import argparse
import json
import os
import warnings
from typing import Dict, List

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from utils.common import (
    build_dataset,
    compute_token_statistics,
    get_data,
    load_txt,
    preprocess_case_data,
    print_token_statistics,
)

warnings.filterwarnings("ignore")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def prepare_cross_encoder_data(dataset_path: str, year: str, segment: str = "test"):
    """
    Prepare data in cross_encoder format for evaluation (GLOBAL CORPUS APPROACH).
    
    WARNING: This creates a global corpus with candidates from ALL cases.
    This may lead to data leakage and unrealistic evaluation.
    
    Returns:
        queries: Dict[case_id, query_text]
        corpus: Dict[candidate_id, candidate_text]
        relevant_docs: Dict[case_id, Set[candidate_id]]
    """
    formatted_data = build_dataset(dataset_path, year, segment)

    queries = {}
    corpus = {}
    relevant_docs = {}
    corpus_counter = 0

    for sample in formatted_data:
        case_id = sample["id"]
        queries[case_id] = sample["text"]
        relevant_docs[case_id] = set()

        # Add positive candidates to corpus
        for pos_cand in sample["pos_candidates"]:
            corpus[corpus_counter] = pos_cand["text"]
            relevant_docs[case_id].add(corpus_counter)
            corpus_counter += 1

        # Add negative candidates to corpus
        for neg_cand in sample["neg_candidates"]:
            corpus[corpus_counter] = neg_cand["text"]
            corpus_counter += 1

    return queries, corpus, relevant_docs


def prepare_cross_encoder_data_per_case(
    dataset_path: str, year: str, segment: str = "test"
):
    evaluation_data = {}
    corpus_dir, cases_dir, label_data = get_data(dataset_path, year, segment)
    
    for case in cases_dir:
        case_id = case
        case_corpus = {}
        case_relevant_docs = set()

        query_text = load_txt(corpus_dir / case / "entailed_fragment.txt")
        
        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))
        
        for cand_case in candidate_cases:
            cand_case_file = candidate_dir / cand_case
            cand_case_data = preprocess_case_data(
                cand_case_file, uncased=False
            )
            case_corpus[cand_case] = cand_case_data
            if cand_case in label_data[case]:
                case_relevant_docs.add(cand_case)

        evaluation_data[case_id] = {
            "query": query_text,
            "corpus": case_corpus,
            "relevant_docs": case_relevant_docs,
        }
    
    return evaluation_data


class CrossEncoderModel:
    """Cross-encoder model wrapper that handles both seq2seq and causal LM models."""
    
    def __init__(self, model_path: str, tokenizer_path: str, model_type: str = "seq2seq", device: str = "cuda"):
        self.device = device
        self.model_type = model_type.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        if self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            self.true_token_id = self.tokenizer.get_vocab().get("▁true", self.tokenizer.get_vocab().get("true"))
            self.false_token_id = self.tokenizer.get_vocab().get("▁false", self.tokenizer.get_vocab().get("false"))
        elif self.model_type == "causallm":
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'seq2seq' or 'causallm'")
    
    def predict_scores(self, queries: List[str], documents: List[str], batch_size: int = 32) -> List[float]:
        """
        Predict relevance scores for query-document pairs.
        
        Args:
            queries: List of query texts
            documents: List of document texts (same length as queries)
            batch_size: Batch size for inference
            
        Returns:
            List of relevance scores (0-1 range)
        """
        if len(queries) != len(documents):
            raise ValueError("Number of queries and documents must be equal")
            
        scores = []
        total_batches = (len(queries) + batch_size - 1) // batch_size
        
        # Only show progress bar for large datasets
        show_progress = len(queries) >= 100
        
        pbar = tqdm(total=total_batches, desc="Predicting scores", disable=not show_progress)
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            
            if self.model_type == "seq2seq":
                batch_scores = self._predict_seq2seq_batch(batch_queries, batch_documents)
            else:  # causallm
                batch_scores = self._predict_causallm_batch(batch_queries, batch_documents)
                
            scores.extend(batch_scores)
            
            if show_progress:
                pbar.update(1)
                # Less frequent logging for very large batches
                batch_num = (i // batch_size) + 1
                if batch_num % max(1, total_batches // 20) == 0:
                    pbar.set_postfix({"batch": f"{batch_num}/{total_batches}", "processed": f"{len(scores):,}"})
        
        if show_progress:
            pbar.close()
            
        return scores
    
    def _predict_seq2seq_batch(self, queries: List[str], documents: List[str]) -> List[float]:
        """Predict scores using seq2seq model (e.g., MonoT5)."""
        try:
            # Lazily cache token ids for efficiency & robustness
            if not hasattr(self, "token_true_id") or self.token_true_id is None:
                # Use the last piece to be safe if it ever splits
                self.token_true_id = self.tokenizer.encode("true", add_special_tokens=False)[-1]
                self.token_false_id = self.tokenizer.encode("false", add_special_tokens=False)[-1]

            # Build prompts
            input_texts = [f"Query: {q} Document: {d} Relevant:" for q, d in zip(queries, documents)]

            # Tokenize
            enc = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,                # pad to longest in batch
                truncation=True,
                return_token_type_ids=False,
            ).to(self.device)

            # Generate exactly ONE token so scores[0] is the first-step logits
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=1,
                    do_sample=False,                    # deterministic (greedy) for scoring
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # First-step logits over vocab: (B, V)
            step_logits = out.scores[0]

            # Keep only {false, true} (order: [false, true]) -> (B, 2)
            two_logits = step_logits[:, [self.token_false_id, self.token_true_id]]

            # Log-softmax over the 2 labels, then take log P(true)
            log_probs = F.log_softmax(two_logits, dim=-1)   # (B,2)
            p_true = log_probs[:, 1].exp()                  # P(true | {false,true})

            return p_true.tolist()
            
        except Exception as e:
            print(f"Error in seq2seq batch prediction: {e}")
            return [0.5] * len(queries)  # Return neutral scores on error
    
    def _predict_causallm_batch(self, queries: List[str], documents: List[str]) -> List[float]:
        """Predict scores using causal LM model (e.g., Qwen, Llama)."""
        try:
            # === 1) Prompt: make the next token be ' yes' or ' no'
            # Use yes/no for causal LMs (widely single-token). Note trailing space!
            input_texts = [
                (
                    f"Query: {q}\n"
                    f"Document: {d}\n"
                    f"Is the document relevant to the query? Answer 'yes' or 'no': "
                )
                for q, d in zip(queries, documents)
            ]

            encoded = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # === 2) Forward pass to get next-token logits at the last position
            with torch.no_grad():
                out = self.model(**encoded)
                logits = out.logits  # (B, T, V)

            attn = encoded["attention_mask"].long()         # (B, T)
            last_pos = attn.sum(dim=1) - 1                  # (B,)
            b_idx = torch.arange(logits.size(0), device=logits.device)
            next_logits = logits[b_idx, last_pos, :]        # (B, V)

            # === 3) Prepare token ids (cache once on the instance for speed)
            if not hasattr(self, "token_yes_id") or self.token_yes_id is None:
                # Leading space ensures we hit the single token variant for GPT/BPE families.
                self.token_yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
                self.token_no_id  = self.tokenizer.encode("no", add_special_tokens=False)[0]

            # Safety: if ids are invalid, bail to neutral
            if self.token_yes_id < 0 or self.token_no_id < 0:
                return [0.5] * len(queries)

            # === 4) Two-class normalize → log P(yes)
            two_logits = torch.stack(
                [next_logits[:, self.token_no_id], next_logits[:, self.token_yes_id]],
                dim=1,  # (B, 2) in order [no, yes]
            )
            log_p = F.log_softmax(two_logits, dim=1)[:, 1]  # log P(yes | {no, yes})
            p_yes = log_p.exp()                              # (B,)

            return p_yes.tolist()

        except Exception as e:
            print(f"Error in causal LM batch prediction: {e}")
            return [0.5] * len(queries)


def run_cross_encoder_inference(
    queries: Dict,
    corpus: Dict,
    model: CrossEncoderModel,
    batch_size: int = 32,
):
    """
    Run cross-encoder inference on queries and corpus (GLOBAL CORPUS APPROACH).
    
    Args:
        queries: Dict[case_id, query_text]
        corpus: Dict[candidate_id, candidate_text]
        model: CrossEncoderModel instance
        batch_size: Batch size for prediction
        
    Returns:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
    """
    # Prepare all query-document pairs
    query_list = []
    document_list = []
    pair_info = []  # (case_id, candidate_id)
    
    for case_id, query_text in queries.items():
        for candidate_id, candidate_text in corpus.items():
            query_list.append(query_text)
            document_list.append(candidate_text)
            pair_info.append((case_id, candidate_id))
    
    # Get predictions for all pairs
    pair_scores = model.predict_scores(query_list, document_list, batch_size)
    
    # Organize results by case and candidate
    scores = {}
    for i, (case_id, candidate_id) in enumerate(pair_info):
        if case_id not in scores:
            scores[case_id] = {}
        scores[case_id][candidate_id] = pair_scores[i]
    
    return scores


def run_cross_encoder_inference_per_case(
    evaluation_data: Dict,
    model: CrossEncoderModel,
    batch_size: int = 32,
):
    """
    Run cross-encoder inference per case (PER-CASE APPROACH).
    
    Each case is evaluated independently with its own candidate pool.
    
    Args:
        evaluation_data: Dict[case_id, {
            "query": query_text,
            "corpus": Dict[candidate_id, candidate_text],
            "relevant_docs": Set[candidate_id]
        }]
        model: CrossEncoderModel instance
        batch_size: Batch size for prediction
        
    Returns:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
    """
    scores = {}
    total_cases = len(evaluation_data)
    
    # Only show progress for datasets with many cases
    show_progress = total_cases >= 10
    
    case_pbar = tqdm(total=total_cases, desc="Processing cases", disable=not show_progress)
    
    for case_idx, (case_id, case_data) in enumerate(evaluation_data.items()):
        query_text = case_data["query"]
        case_corpus = case_data["corpus"]
        
        if not case_corpus:
            scores[case_id] = {}
            continue
        
        # Prepare query-document pairs for this case
        query_list = []
        document_list = []
        candidate_ids = []
        
        for candidate_id, candidate_text in case_corpus.items():
            query_list.append(query_text)
            document_list.append(candidate_text)
            candidate_ids.append(candidate_id)
        
        # Get predictions for this case
        case_scores = model.predict_scores(query_list, document_list, batch_size)
        
        # Store results
        scores[case_id] = {}
        for i, candidate_id in enumerate(candidate_ids):
            scores[case_id][candidate_id] = case_scores[i]
        
        if show_progress:
            case_pbar.update(1)
            # Update description every 10% of cases
            if (case_idx + 1) % max(1, total_cases // 10) == 0:
                case_pbar.set_postfix({"case": f"{case_idx + 1}/{total_cases}"})
    
    if show_progress:
        case_pbar.close()
        
    return scores


def calculate_recall_at_k(scores: Dict, relevant_docs: Dict, k: int) -> float:
    """
    Calculate recall@k for cross-encoder results.
    
    Args:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
        relevant_docs: Dict[case_id, Set[candidate_id]]
        k: Number of top predictions to consider
        
    Returns:
        micro_recall: Micro-averaged recall@k
    """
    total_relevant_retrieved = 0
    total_relevant_items = 0

    for case_id in scores:
        if case_id not in relevant_docs or not relevant_docs[case_id]:
            continue

        # Get top-k predictions for this case
        case_scores = scores[case_id]
        top_k_candidates = sorted(
            case_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]
        top_k_ids = {candidate_id for candidate_id, _ in top_k_candidates}

        # Get ground truth for this case
        relevant_candidates = relevant_docs[case_id]
        print(top_k_ids)
        print(relevant_candidates)
        # Calculate recall for this case
        retrieved_relevant = top_k_ids & relevant_candidates
        total_relevant_retrieved += len(retrieved_relevant)
        total_relevant_items += len(relevant_candidates)

    # Calculate overall recall@k
    micro_recall = (
        total_relevant_retrieved / total_relevant_items
        if total_relevant_items > 0
        else 0
    )
    return micro_recall


def calculate_precision_at_k(scores: Dict, relevant_docs: Dict, k: int) -> float:
    """
    Calculate precision@k for cross-encoder results.
    
    Args:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
        relevant_docs: Dict[case_id, Set[candidate_id]]
        k: Number of top predictions to consider
        
    Returns:
        micro_precision: Micro-averaged precision@k
    """
    total_retrieved = 0
    total_relevant_retrieved = 0

    for case_id in scores:
        if case_id not in relevant_docs or not relevant_docs[case_id]:
            continue

        # Get top-k predictions for this case
        case_scores = scores[case_id]
        top_k_candidates = sorted(
            case_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]
        top_k_ids = {candidate_id for candidate_id, _ in top_k_candidates}

        # Get ground truth for this case
        relevant_candidates = relevant_docs[case_id]

        # Calculate precision components
        retrieved_relevant = top_k_ids & relevant_candidates
        total_retrieved += len(top_k_ids)
        total_relevant_retrieved += len(retrieved_relevant)

    # Calculate overall precision@k
    micro_precision = (
        total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    )
    return micro_precision


def calculate_f1_at_k(scores: Dict, relevant_docs: Dict, k: int) -> float:
    """
    Calculate F1@k for cross-encoder results.
    
    Args:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
        relevant_docs: Dict[case_id, Set[candidate_id]]
        k: Number of top predictions to consider
        
    Returns:
        micro_f1: Micro-averaged F1@k
    """
    recall = calculate_recall_at_k(scores, relevant_docs, k)
    precision = calculate_precision_at_k(scores, relevant_docs, k)

    if recall + precision > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1


def run_eval_cross_encoder(scores: Dict, relevant_docs: Dict):
    """
    Run comprehensive evaluation on cross-encoder results (GLOBAL CORPUS APPROACH).
    
    Args:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
        relevant_docs: Dict[case_id, Set[candidate_id]]
        
    Returns:
        results: Dict with recall, precision, and F1 scores at different k values
    """
    eval_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []

    for k in eval_list:
        recall_score_list.append(calculate_recall_at_k(scores, relevant_docs, k))
        precision_score_list.append(calculate_precision_at_k(scores, relevant_docs, k))
        f1_score_list.append(calculate_f1_at_k(scores, relevant_docs, k))

    results = {
        "eval_list": eval_list,
        "recall_score_list": recall_score_list,
        "precision_score_list": precision_score_list,
        "f1_score_list": f1_score_list,
    }
    return results


def run_eval_cross_encoder_per_case(scores: Dict, evaluation_data: Dict):
    """
    Run comprehensive evaluation on cross-encoder results (PER-CASE APPROACH).
    
    Args:
        scores: Dict[case_id, Dict[candidate_id, relevance_score]]
        evaluation_data: Dict[case_id, {"query": str, "corpus": Dict, "relevant_docs": Set}]
        
    Returns:
        results: Dict with recall, precision, and F1 scores at different k values
    """
    eval_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []

    # Extract relevant_docs from evaluation_data
    relevant_docs = {}
    for case_id, case_data in evaluation_data.items():
        relevant_docs[case_id] = case_data["relevant_docs"]

    for k in eval_list:
        recall_score_list.append(calculate_recall_at_k(scores, relevant_docs, k))
        precision_score_list.append(calculate_precision_at_k(scores, relevant_docs, k))
        f1_score_list.append(calculate_f1_at_k(scores, relevant_docs, k))

    results = {
        "eval_list": eval_list,
        "recall_score_list": recall_score_list,
        "precision_score_list": precision_score_list,
        "f1_score_list": f1_score_list,
    }
    return results


def eval_cross_encoder_comprehensive(
    dataset_path: str,
    year: str,
    segment: str,
    model_path: str,
    tokenizer_path: str,
    model_type: str,
    out_path: str,
    device: str = "cuda",
    approach: str = "per_case",  # "global" or "per_case"
    batch_size: int = 16,
):
    """
    Complete evaluation pipeline for cross-encoder model with switchable approaches.
    
    Args:
        dataset_path: Path to dataset
        year: Dataset year
        segment: Data segment (train/dev/test)
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        model_type: Model type ("seq2seq" or "causallm")
        out_path: Output file path
        device: Device to run on
        approach: "global" for global corpus, "per_case" for per-case evaluation
        batch_size: Batch size for inference
    """
    # Load model
    model = CrossEncoderModel(model_path, tokenizer_path, model_type, device)

    if approach == "global":
        return _eval_global_corpus_cross(dataset_path, year, segment, model, out_path, batch_size)
    elif approach == "per_case":
        return _eval_per_case_cross(dataset_path, year, segment, model, out_path, batch_size)
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'global' or 'per_case'")


def _eval_global_corpus_cross(dataset_path, year, segment, model, out_path, batch_size):
    """Global corpus evaluation approach for cross-encoder."""
    print("Using GLOBAL CORPUS approach")
    print("WARNING: This may lead to data leakage and unrealistic evaluation.")

    # Prepare data
    print("Preparing data...")
    queries, corpus, relevant_docs = prepare_cross_encoder_data(
        dataset_path, year, segment
    )

    print(f"Loaded {len(queries)} queries, {len(corpus)} corpus documents")
    print(
        f"Average relevant docs per query: {sum(len(docs) for docs in relevant_docs.values()) / len(queries):.2f}"
    )

    # Run inference
    print("Running cross-encoder inference...")
    print(f"Total query-document pairs to process: {len(queries) * len(corpus):,}")
    scores = run_cross_encoder_inference(
        queries, corpus, model, batch_size=batch_size
    )

    # Evaluate
    print("Evaluating results...")
    results = run_eval_cross_encoder(scores, relevant_docs)

    # Save results
    results["approach"] = "global_corpus"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def _eval_per_case_cross(dataset_path, year, segment, model, out_path, batch_size):
    """Per-case evaluation approach for cross-encoder."""
    print("Using PER-CASE approach")
    print("Each case searches only within its own candidate pool.")

    # Prepare data
    print("Preparing data...")
    evaluation_data = prepare_cross_encoder_data_per_case(dataset_path, year, segment)

    total_queries = len(evaluation_data)
    total_corpus = sum(
        len(case_data["corpus"]) for case_data in evaluation_data.values()
    )
    avg_candidates = total_corpus / total_queries if total_queries > 0 else 0
    avg_relevant = (
        sum(len(case_data["relevant_docs"]) for case_data in evaluation_data.values())
        / total_queries
        if total_queries > 0
        else 0
    )

    print(f"Loaded {total_queries} queries, {total_corpus} total corpus documents")
    print(f"Average candidates per case: {avg_candidates:.2f}")
    print(f"Average relevant docs per query: {avg_relevant:.2f}")

    # Compute and display token statistics
    print("Computing token statistics...")
    try:
        token_stats = compute_token_statistics(
            dataset_path=dataset_path,
            year=year,
            segment=segment,
            tokenizer_name=model.tokenizer.name_or_path,
            threshold=512,
        )
        print_token_statistics(token_stats, label="Cross-Encoder Token Statistics (512-token truncation)")
    except Exception as e:
        print(f"Warning: Could not compute token statistics: {e}")

    # Run inference
    print("Running cross-encoder inference per case...")
    total_pairs = sum(len(case_data["corpus"]) for case_data in evaluation_data.values())
    print(f"Total query-document pairs to process: {total_pairs:,}")
    scores = run_cross_encoder_inference_per_case(
        evaluation_data, model, batch_size=batch_size
    )

    with open(f"scores_{out_path}", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # Evaluate
    print("Evaluating results...")
    results = run_eval_cross_encoder_per_case(scores, evaluation_data)

    # Save results
    # results["approach"] = "per_case"
    # results["stats"] = {
    #     "total_cases": total_queries,
    #     "total_corpus_docs": total_corpus,
    #     "avg_candidates_per_case": avg_candidates,
    #     "avg_relevant_per_case": avg_relevant,
    # }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return scores


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cross-encoder inference script")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--pretrained_model", type=str, required=True,
                       help="Pretrained model name or path")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                       help="Path to tokenizer")
    parser.add_argument("--year", type=str, required=True,
                       help="Dataset year")
    parser.add_argument("--model_type", type=str, default="seq2seq",
                       choices=["seq2seq", "causallm"],
                       help="Model type")
    parser.add_argument("--dynamic_sampling_strategy", type=str, default="hard",
                       choices=["hard", "random"],
                       help="Dynamic sampling strategy")
    parser.add_argument("--max_negatives_per_positive", type=int, default=10,
                       help="Maximum negatives per positive")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--dataset_path", type=str, default="dataset",
                       help="Path to dataset")
    parser.add_argument("--segment", type=str, default="test",
                       choices=["train", "dev", "test"],
                       help="Dataset segment to evaluate")
    parser.add_argument("--approach", type=str, default="per_case",
                       choices=["global", "per_case"],
                       help="Evaluation approach")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Configuration from command line args
    dataset_path = args.dataset_path
    year = args.year
    segment = args.segment
    model_path = args.checkpoint_path
    tokenizer_path = args.tokenizer_path
    model_type = args.model_type
    device = args.device
    approach = args.approach
    dynamic_sampling_strategy = args.dynamic_sampling_strategy
    batch_size = 24

    # Generate output filename
    model_base_name = args.pretrained_model.split('/')[-1]
    out_path = f"{model_base_name}_{year}_cross_{model_type}_{args.dynamic_sampling_strategy}_neg{args.max_negatives_per_positive}.json"

    print(f"Starting cross-encoder evaluation with {approach} approach...")
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset_path}, Year: {year}, Segment: {segment}")
    print(f"Results will be saved to: {out_path}")

    eval_cross_encoder_comprehensive(
        dataset_path=dataset_path,
        year=year,
        segment=segment,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        model_type=model_type,
        out_path=out_path,
        device=device,
        approach=approach,
        batch_size=batch_size,
    )

    print(f"\nEvaluation completed! Results saved to: {out_path}")
    print("Done")