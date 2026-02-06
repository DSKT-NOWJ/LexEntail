import argparse
import json
import os
import warnings
from typing import Dict

from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm

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


# def model_encoding(query: str):
#     query_embedding = model.encode(query, device=device, show_progress_bar = False)
#     return query_embedding

# def batch_encoding(query_list: list):
#     query_embedding = model.encode(query_list, device=device, show_progress_bar = False)
#     return query_embedding.tolist()


def prepare_bi_encoder_data(dataset_path: str, year: str, segment: str = "test"):
    """
    Prepare data in bi_encoder format for evaluation (GLOBAL CORPUS APPROACH).

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


def prepare_bi_encoder_data_per_case(
    dataset_path: str, year: str, segment: str = "test"
):
    """
    Prepare data in bi_encoder format for evaluation (PER-CASE APPROACH).

    Each case only searches within its own candidate pool - more realistic evaluation.
    This matches the actual task: find relevant paragraphs within THIS case's documents.

    Returns:
        evaluation_data: Dict[case_id, {
            "query": query_text,
            "corpus": Dict[candidate_id, candidate_text],
            "relevant_docs": Set[candidate_id]
        }]
    """
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
            cand_case_data = preprocess_case_data(cand_case_file, uncased=False)
            case_corpus[cand_case] = cand_case_data
            if cand_case in label_data[case]:
                case_relevant_docs.add(cand_case)

        evaluation_data[case_id] = {
            "query": query_text,
            "corpus": case_corpus,
            "relevant_docs": case_relevant_docs,
        }

    return evaluation_data


def run_bi_encoder_inference(
    queries: Dict,
    corpus: Dict,
    model: SentenceTransformer,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Run bi-encoder inference on queries and corpus (GLOBAL CORPUS APPROACH).

    Args:
        queries: Dict[case_id, query_text]
        corpus: Dict[candidate_id, candidate_text]
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
    """
    # Encode all queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )

    # Encode all corpus texts
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )

    # Calculate similarity scores
    scores = {}
    for i, query_id in enumerate(tqdm(query_ids, desc="Calculating similarities")):
        query_embedding = query_embeddings[i : i + 1]  # Keep batch dimension
        similarities = model.predict_scores(query_embedding, corpus_embeddings)[
            0
        ]  # Remove batch dimension

        # Store scores for this query
        scores[query_id] = {}
        for j, corpus_id in enumerate(corpus_ids):
            scores[query_id][corpus_id] = float(similarities[j])

    return scores


def run_bi_encoder_inference_per_case(
    evaluation_data: Dict,
    model: SentenceTransformer,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Run bi-encoder inference per case (PER-CASE APPROACH).

    Each case is evaluated independently with its own candidate pool.

    Args:
        evaluation_data: Dict[case_id, {
            "query": query_text,
            "corpus": Dict[candidate_id, candidate_text],
            "relevant_docs": Set[candidate_id]
        }]
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        device: Device to run on

    Returns:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
    """
    scores = {}

    for case_id, case_data in tqdm(evaluation_data.items(), desc="Processing cases"):
        query_text = case_data["query"]
        case_corpus = case_data["corpus"]

        if not case_corpus:
            scores[case_id] = {}
            continue

        query_embedding = model.encode_query(query_text, convert_to_tensor=True)

        # Encode case-specific corpus
        corpus_ids = list(case_corpus.keys())
        corpus_texts = [case_corpus[cid] for cid in corpus_ids]

        corpus_embeddings = model.encode_document(corpus_texts, convert_to_tensor=True)

        # Calculate similarities for this case only
        similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]

        # Store scores for this case
        scores[case_id] = {}
        for j, corpus_id in enumerate(corpus_ids):
            scores[case_id][corpus_id] = float(similarity_scores[j])

    return scores


def calculate_recall(predict, label):
    n_true = 0
    for i in label:
        if i in predict:
            n_true += 1
    return n_true


def calculate_precision(predict, label):
    if len(predict) == 0:
        return 0
    n_true = 0
    for i in predict:
        if i in label:
            n_true += 1
    return n_true / len(predict)


def calculate_recall_at_k(scores: Dict, relevant_docs: Dict, k: int) -> float:
    """
    Calculate recall@k for bi-encoder results.

    Args:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
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
    Calculate precision@k for bi-encoder results.

    Args:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
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
    Calculate F1@k for bi-encoder results.

    Args:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
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


def run_eval_bi_encoder(scores: Dict, relevant_docs: Dict):
    """
    Run comprehensive evaluation on bi-encoder results (GLOBAL CORPUS APPROACH).

    Args:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
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


def run_eval_bi_encoder_per_case(scores: Dict, evaluation_data: Dict):
    """
    Run comprehensive evaluation on bi-encoder results (PER-CASE APPROACH).

    Args:
        scores: Dict[case_id, Dict[candidate_id, similarity_score]]
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


def eval_bi_encoder_comprehensive(
    dataset_path: str,
    year: str,
    segment: str,
    model_name: str,
    out_path: str,
    device: str = "cuda",
    approach: str = "global",  # "global" or "per_case"
):
    """
    Complete evaluation pipeline for bi-encoder model with switchable approaches.

    Args:
        dataset_path: Path to dataset
        year: Dataset year
        segment: Data segment (train/dev/test)
        model_name: Model name or path
        out_path: Output file path
        device: Device to run on
        approach: "global" for global corpus, "per_case" for per-case evaluation
    """
    # Load model
    model = SentenceTransformer(model_name, device=device)

    if approach == "global":
        return _eval_global_corpus(dataset_path, year, segment, model, out_path, device)
    elif approach == "per_case":
        return _eval_per_case(dataset_path, year, segment, model, out_path, device)
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'global' or 'per_case'")


def _eval_global_corpus(dataset_path, year, segment, model, out_path, device):
    """Global corpus evaluation approach."""
    print("Using GLOBAL CORPUS approach")
    print("WARNING: This may lead to data leakage and unrealistic evaluation.")

    # Prepare data
    print("Preparing data...")
    queries, corpus, relevant_docs = prepare_bi_encoder_data(
        dataset_path, year, segment
    )

    print(f"Loaded {len(queries)} queries, {len(corpus)} corpus documents")
    print(
        f"Average relevant docs per query: {sum(len(docs) for docs in relevant_docs.values()) / len(queries):.2f}"
    )

    # Run inference
    print("Running bi-encoder inference...")
    scores = run_bi_encoder_inference(
        queries, corpus, model, batch_size=32, device=device
    )

    # Evaluate
    print("Evaluating results...")
    results = run_eval_bi_encoder(scores, relevant_docs)

    # Save results
    results["approach"] = "global_corpus"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print results
    # _print_results(
    #     results,
    #     dataset_path,
    #     year,
    #     segment,
    #     model.model_name_or_path,
    #     device,
    #     "GLOBAL CORPUS",
    # )
    return results


def _eval_per_case(dataset_path, year, segment, model, out_path, device):
    """Per-case evaluation approach."""
    print("Using PER-CASE approach")
    print("Each case searches only within its own candidate pool.")

    # Prepare data
    print("Preparing data...")
    evaluation_data = prepare_bi_encoder_data_per_case(dataset_path, year, segment)

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
            tokenizer_name=model.model_name_or_path,
            threshold=512,
        )
        print_token_statistics(token_stats, label="Bi-Encoder Token Statistics (512-token truncation)")
    except Exception as e:
        print(f"Warning: Could not compute token statistics: {e}")

    # Run inference
    print("Running bi-encoder inference per case...")
    scores = run_bi_encoder_inference_per_case(
        evaluation_data, model, batch_size=32, device=device
    )

    with open(f"scores_{out_path}", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # Evaluate
    print("Evaluating results...")
    results = run_eval_bi_encoder_per_case(scores, evaluation_data)

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

    # Print results
    # _print_results(
    #     results,
    #     dataset_path,
    #     year,
    #     segment,
    #     model.model_name_or_path,
    #     device,
    #     "PER-CASE",
    # )
    return scores


def _print_results(
    results, dataset_path, year, segment, model_name, device, approach_name
):
    """Helper function to print formatted results."""
    print("\n" + "=" * 70)
    print(f"BI-ENCODER EVALUATION RESULTS ({approach_name})")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Year: {year}, Segment: {segment}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Approach: {approach_name}")

    if "stats" in results:
        stats = results["stats"]
        print(f"Total cases: {stats['total_cases']}")
        print(f"Total corpus docs: {stats['total_corpus_docs']}")
        print(f"Avg candidates per case: {stats['avg_candidates_per_case']:.2f}")
        print(f"Avg relevant per case: {stats['avg_relevant_per_case']:.2f}")

    print("-" * 70)

    eval_list = results["eval_list"]
    recall_scores = results["recall_score_list"]
    precision_scores = results["precision_score_list"]
    f1_scores = results["f1_score_list"]

    print(f"{'k':<5} {'Recall@k':<10} {'Precision@k':<12} {'F1@k':<10}")
    print("-" * 40)
    for i, k in enumerate(eval_list):
        print(
            f"{k:<5} {recall_scores[i]:<10.4f} {precision_scores[i]:<12.4f} {f1_scores[i]:<10.4f}"
        )

    print("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bi-encoder inference script")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument("--year", type=str, required=True, help="Dataset year")
    parser.add_argument(
        "--dynamic_sampling_strategy",
        type=str,
        default="random",
        help="Dynamic sampling strategy",
    )
    parser.add_argument(
        "--max_negatives_per_positive",
        type=int,
        default=10,
        help="Maximum negatives per positive",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="dataset", help="Path to dataset"
    )
    parser.add_argument(
        "--segment",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset segment to evaluate",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="per_case",
        choices=["global", "per_case"],
        help="Evaluation approach",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configuration from command line args
    dataset_path = args.dataset_path
    year = args.year
    segment = args.segment
    model_name = args.checkpoint_path
    device = args.device
    approach = args.approach
    dynamic_sampling_strategy = args.dynamic_sampling_strategy

    # Generate output filename
    model_base_name = args.pretrained_model.split("/")[-1]
    out_path = f"{model_base_name}_{year}_{dynamic_sampling_strategy}_neg{args.max_negatives_per_positive}.json"

    print(f"Starting bi-encoder evaluation with {approach} approach...")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}, Year: {year}, Segment: {segment}")
    print(f"Results will be saved to: {out_path}")

    eval_bi_encoder_comprehensive(
        dataset_path=dataset_path,
        year=year,
        segment=segment,
        model_name=model_name,
        out_path=out_path,
        device=device,
        approach=approach,
    )

    print(f"\nEvaluation completed! Results saved to: {out_path}")
    print("Done")
