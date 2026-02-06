#!/usr/bin/env python3
"""
Fusion analysis script adapted for LCE framework.

This script integrates with existing inference scripts (infer_bi.py, infer_cross.py)
and BM25 implementation to perform fusion analysis with different normalization methods.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from itertools import combinations_with_replacement

import numpy as np
import torch

from bm25 import predict_all_bm25
from infer_bi import (
    eval_bi_encoder_comprehensive,
)
from infer_cross import (
    eval_cross_encoder_comprehensive,
)
from utils.common import get_data, load_json

root = Path(__file__).resolve().parents[1]


def load_model_results(results_dir: str, model_type: str, year: str) -> Dict:
    """
    Load inference results from JSON files.

    Args:
        results_dir: Directory containing result files
        model_type: Type of model ('bi' or 'cross')
        year: Dataset year

    Returns:
        Dict of model results
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    # Find all result files matching the pattern
    pattern = f"*{year}*{model_type}*.json"
    for result_file in results_path.glob(pattern):
        try:
            with open(result_file, "r") as f:
                result_data = json.load(f)

            # Extract model name from filename
            model_name = result_file.stem
            results[model_name] = result_data
        except Exception:
            pass

    return results


def run_bm25_inference(dataset_path: str, year: str, segment: str = "test") -> Dict:
    """
    Run BM25 inference using existing BM25 implementation.

    Args:
        dataset_path: Path to dataset
        year: Dataset year
        segment: Data segment (train/dev/test)

    Returns:
        BM25 scores in the format: {case_id: {candidate_id: score}}
    """
    bm25_index_path = str(root / f"dataset/bm25_indexes/coliee_task2/{year}/{segment}")

    if not os.path.exists(bm25_index_path):
        return {}

    bm25_scores = predict_all_bm25(
        dataset_path=dataset_path,
        year=year,
        bm25_index_path=bm25_index_path,
        eval_segment=segment,
    )

    return bm25_scores


def run_bi_encoder_inference(
    checkpoint_path: str,
    pretrained_model: str,
    year: str,
    dataset_path: str,
    out_path: str,
    segment: str = "test",
    device: str = "cuda",
) -> Dict:
    """
    Run bi-encoder inference using existing implementation.

    Returns:
        Scores in the format expected by fusion: {case_id: {candidate_id: score}}
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        return {}

        # Run evaluation using existing function
    results = eval_bi_encoder_comprehensive(
        dataset_path=dataset_path,
        year=year,
        segment=segment,
        model_name=checkpoint_path,
        out_path=out_path,
        device=device,
        approach="per_case",
    )
    return results


def run_cross_encoder_inference(
    checkpoint_path: str,
    pretrained_model: str,
    tokenizer_path: str,
    model_type: str,
    year: str,
    out_path: str,
    dataset_path: str,
    segment: str = "test",
    device: str = "cuda",
) -> Dict:
    """
    Run cross-encoder inference using existing implementation.

    Returns:
        Scores in the format expected by fusion: {case_id: {candidate_id: score}}
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        return {}

    # Run evaluation using existing function
    results = eval_cross_encoder_comprehensive(
        dataset_path=dataset_path,
        year=year,
        segment=segment,
        model_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        model_type=model_type,
        out_path=out_path,
        device=device,
        approach="per_case",
    )

    return results


class Aggregator:
    """
    A class for aggregating ranked lists using different fusion methods.
    """

    @classmethod
    def fuse(
        cls,
        ranked_lists: dict[str, dict[str, dict[str, float]]],
        method: str,
        normalization: str = None,
        linear_weights: dict[str, float] = None,
        percentile_distributions: dict[str, np.array] = None,
    ) -> dict[str, list[dict[str, any]]]:
        """
        Fuse the ranked lists of different retrieval systems.

        Args:
            ranked_lists: Dict[system_name, Dict[case_id, Dict[candidate_id, score]]]
            method: The fusion method to use ('bcf', 'rrf', 'nsf')
            normalization: The normalization method to use
            linear_weights: Weights for linear combination (NSF)
            percentile_distributions: Percentile distributions for normalization
            return_topk: Number of top results to return

        Returns:
            Fused results: Dict[case_id, List[Dict[candidate_id, score]]]
        """
        # Get all case IDs from the first system
        case_ids = list(next(iter(ranked_lists.values())).keys())

        # Validate inputs
        assert all(
            set(system_results.keys()) == set(case_ids)
            for system_results in ranked_lists.values()
        ), "Not all systems have results for the same cases."

        if method == "nsf" and linear_weights:
            assert set(ranked_lists.keys()) == set(linear_weights.keys()), (
                "System names in linear_weights don't match ranked_lists keys."
            )

        final_results = {}

        for case_id in case_ids:
            case_fusion_results = []

            for system, system_results in ranked_lists.items():
                case_scores = system_results.get(case_id, {})

                if not case_scores:
                    continue

                if method == "bcf":
                    case_scores = cls.transform_scores(
                        case_scores, transformation="borda-count"
                    )
                elif method == "rrf":
                    case_scores = cls.transform_scores(
                        case_scores, transformation="reciprocal-rank"
                    )
                elif method == "nsf":
                    case_scores = cls.transform_scores(
                        case_scores,
                        transformation=normalization,
                        percentile_distr=(
                            percentile_distributions.get(system)
                            if percentile_distributions
                            else None
                        ),
                    )
                    if linear_weights:
                        case_scores = cls.weight_scores(
                            case_scores, w=linear_weights[system]
                        )

                case_fusion_results.append(case_scores)

            # Aggregate scores for this case
            final_results[case_id] = cls.aggregate_scores(*case_fusion_results)

        return final_results

    @staticmethod
    def transform_scores(
        results: dict[str, float],
        transformation: str,
        percentile_distr: np.array = None,
    ) -> dict[str, float]:
        """
        Transform the scores of results using various normalization methods.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if transformation == "borda-count":
            # Sort by original scores (descending) first
            sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
            num_candidates = len(sorted_items)
            return {
                pid: (num_candidates - idx + 1) / num_candidates
                for idx, (pid, _) in enumerate(sorted_items)
            }

        elif transformation == "reciprocal-rank":
            # Sort by original scores (descending) first
            sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
            k = 60
            return {pid: 1 / (k + idx + 1) for idx, (pid, _) in enumerate(sorted_items)}

        elif transformation == "min-max":
            scores = torch.tensor(
                list(results.values()), device=device, dtype=torch.float32
            )
            min_val, max_val = torch.min(scores), torch.max(scores)
            scores = (
                (scores - min_val) / (max_val - min_val)
                if min_val != max_val
                else torch.ones_like(scores)
            )
            return {
                pid: float(score)
                for pid, score in zip(results.keys(), scores.cpu().numpy())
            }

        elif transformation == "z-score":
            scores = torch.tensor(
                list(results.values()), device=device, dtype=torch.float32
            )
            mean_val, std_val = torch.mean(scores), torch.std(scores)
            scores = (
                (scores - mean_val) / std_val
                if std_val != 0
                else torch.zeros_like(scores)
            )
            return {
                pid: float(score)
                for pid, score in zip(results.keys(), scores.cpu().numpy())
            }

        elif transformation == "percentile-rank":
            if percentile_distr is None:
                return results

            scores = torch.tensor(
                list(results.values()), device=device, dtype=torch.float32
            )
            distribution = torch.tensor(
                percentile_distr, device=device, dtype=torch.float32
            )
            differences = torch.abs(distribution[:, None] - scores)
            scores = torch.argmin(differences, axis=0) / distribution.size(0)
            return {
                pid: float(score)
                for pid, score in zip(results.keys(), scores.cpu().numpy())
            }

        return results

    @staticmethod
    def weight_scores(results: dict[str, float], w: float) -> dict[str, float]:
        """Weight the scores by a constant factor."""
        return {candidate_id: score * w for candidate_id, score in results.items()}

    @staticmethod
    def aggregate_scores(*args: dict[str, float]) -> List[Dict]:
        """
        Aggregate scores from multiple systems.

        Returns:
            Sorted list of {candidate_id, score} dicts
        """
        agg_results = defaultdict(float)
        for results in args:
            if results:  # Check if results is not empty
                for candidate_id, score in results.items():
                    agg_results[candidate_id] += score

        agg_results = sorted(agg_results.items(), key=lambda x: x[0], reverse=False)
        return {candidate_id: float(score) for candidate_id, score in agg_results}


def generate_weight_combinations(
    num_models: int, resolution: float = 0.1
) -> List[List[float]]:
    """
    Generate all possible weight combinations for NSF that sum to 1.0.

    Args:
        num_models: Number of models to generate weights for
        resolution: Step size for weight values (default 0.1)

    Returns:
        List of weight combinations, each summing to 1.0
    """
    # Convert resolution to avoid floating point precision issues
    steps = int(1.0 / resolution)

    weight_combinations = []

    # Generate all combinations where weights sum to 1.0
    for combo in combinations_with_replacement(range(steps + 1), num_models):
        if sum(combo) == steps:
            weights = [x / steps for x in combo]
            # Add all permutations of this combination
            from itertools import permutations

            for perm in set(permutations(weights)):
                weight_combinations.append(list(perm))

    # Remove duplicates and sort
    unique_combinations = []
    for combo in weight_combinations:
        rounded_combo = [round(w, 3) for w in combo]
        if rounded_combo not in unique_combinations:
            unique_combinations.append(rounded_combo)

    return sorted(unique_combinations)


def evaluate_fusion_results(
    fused_results: Dict, relevant_docs: Dict, k_values: List[int] = None
) -> Dict:
    """
    Evaluate fusion results using standard IR metrics.

    Args:
        fused_results: Dict[case_id, List[Dict[candidate_id, score]]]
        relevant_docs: Dict[case_id, Set[candidate_id]]
        k_values: List of k values for evaluation

    Returns:
        Dict with evaluation metrics
    """
    if k_values is None:
        k_values = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]

    eval_results = {}

    for k in k_values:
        total_relevant_retrieved = 0
        total_relevant = 0
        total_retrieved = 0

        for case_id in fused_results:
            if case_id not in relevant_docs:
                continue

            # Get top-k predictions
            predictions = fused_results[case_id][:k]
            predicted_ids = set(item["candidate_id"] for item in predictions)

            # Get ground truth
            ground_truth = set(relevant_docs[case_id])

            # Calculate metrics
            retrieved_relevant = predicted_ids & ground_truth
            total_relevant_retrieved += len(retrieved_relevant)
            total_relevant += len(ground_truth)
            total_retrieved += len(predicted_ids)

        # Calculate recall@k and precision@k
        recall_k = (
            total_relevant_retrieved / total_relevant if total_relevant > 0 else 0
        )
        precision_k = (
            total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        )
        f1_k = (
            2 * precision_k * recall_k / (precision_k + recall_k)
            if (precision_k + recall_k) > 0
            else 0
        )
        f2_k = (
            5 * precision_k * recall_k / (4 * precision_k + recall_k)
            if (precision_k + recall_k) > 0
            else 0
        )

        eval_results[f"recall@{k}"] = recall_k
        eval_results[f"precision@{k}"] = precision_k
        eval_results[f"f1@{k}"] = f1_k
        eval_results[f"f2@{k}"] = f2_k

    return eval_results


def main(args):
    """Main execution function."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all inference results
    all_results = {}

    # Run BM25 if requested
    if args.run_bm25:
        bm25_results = run_bm25_inference(
            dataset_path=args.dataset_path, year=args.year, segment=args.data_split
        )
        if bm25_results:
            all_results["bm25"] = bm25_results

    # Run bi-encoder models if requested
    bi_models = []
    if args.run_bert:
        bi_models.append(("google-bert/bert-base-multilingual-cased", "mbert"))
    if args.run_xlm_roberta:
        bi_models.append(("FacebookAI/xlm-roberta-large", "roberta_large"))
    if args.run_e5:
        bi_models.append(("intfloat/multilingual-e5-large", "e5_large"))
    if args.run_bge:
        bi_models.append(("BAAI/bge-m3", "bge_m3"))

    # for file in os.listdir("./"):
    #     if file.endswith(".json") and file.startswith("scores_"):
    #         for pretrained_model, model_name in bi_models:
    #             if pretrained_model.split('/')[-1] in file:
    #                 print(f"\n{sep}\n# Loading Bi-encoder results: {model_name}\n{sep}")
    #                 with open(file, "r") as f:
    #                     all_results[model_name] = json.load(f)
    #                 break
    #         continue

    for pretrained_model, model_name in bi_models:
        for sampling_strategy in ["hard", "random", "top20"]:
            for neg in ["3", "5"]:
                path = f"scores_{pretrained_model.split('/')[-1]}_{args.year}_{sampling_strategy}_neg{neg}.json"
                if os.path.exists(path):
                    with open(path, "r") as f:
                        all_results[model_name] = json.load(f)
                    break

            # Run inference for models without existing scores
    # for pretrained_model, model_name in bi_models:
    #     if model_name not in all_results:
    #         print(f"\n{sep}\n# Running Bi-encoder inference: {model_name}\n{sep}")
    #         checkpoint_path = f"checkpoints/{pretrained_model.split('/')[-1]}"
    #         if not os.path.exists(checkpoint_path):
    #             print(f"Warning: Checkpoint not found at {checkpoint_path} and no score file available")
    #             continue

    #         out_path = f"scores_{pretrained_model.split('/')[-1]}_{args.year}_{args.dynamic_sampling_strategy}_neg{args.max_negatives_per_positive}.json"
    #         bi_results = run_bi_encoder_inference(
    #             checkpoint_path=checkpoint_path,
    #             pretrained_model=pretrained_model,
    #             year=args.year,
    #             out_path=out_path,
    #             dataset_path=args.dataset_path,
    #             segment=args.data_split,
    #             device=args.device,
    #         )
    #         if bi_results:
    #             all_results[model_name] = bi_results

    # Run cross-encoder models if requested
    cross_models = []
    if args.run_monot5:
        cross_models.append(("castorini/monot5-base", "monot5_base"))
    if args.run_monot5_3b:
        cross_models.append(("castorini/monot5-3b", "monot5_3b"))

    for pretrained_model, model_name in cross_models:
        for sampling_strategy in ["hard", "random", "top20"]:
            for neg in ["3", "5"]:
                path = f"scores_{pretrained_model.split('/')[-1]}_{args.year}_{sampling_strategy}_neg{neg}.json"
                if os.path.exists(path):
                    with open(path, "r") as f:
                        all_results[model_name] = json.load(f)
                    break

        # Fallback: Check for older naming pattern (without _cross_seq2seq)
        # old_out_path = f"{pretrained_model.split('/')[-1]}_{args.year}_{args.dynamic_sampling_strategy}_neg{args.max_negatives_per_positive}.json"
        # if os.path.exists(f"scores_{old_out_path}"):
        #     print(f"Found existing score file (old format): scores_{old_out_path}")
        #     with open(f"scores_{old_out_path}", "r") as f:
        #         all_results[model_name] = json.load(f)
        #     continue

        # # Only check checkpoint if we need to run inference
        # checkpoint_path = f"checkpoints/{pretrained_model.split('/')[-1]}"
        # if not os.path.exists(checkpoint_path):
        #     print(f"Warning: Checkpoint not found at {checkpoint_path} and no score file available")
        #     continue

        # cross_results = run_cross_encoder_inference(
        #         checkpoint_path=checkpoint_path,
        #         pretrained_model=pretrained_model,
        #         tokenizer_path=pretrained_model,
        #         model_type="seq2seq",
        #         year=args.year,
        #         out_path=file,
        #         dataset_path=args.dataset_path,
        #         segment=args.data_split,
        #         device=args.device,
        #     )
        # if cross_results:
        #     all_results[model_name] = cross_results

    if not all_results:
        return

    # Check if we have only one model - skip fusion
    if len(all_results) == 1:
        model_name = list(all_results.keys())[0]
        model_results = all_results[model_name]

        # Convert single model results to the format expected by evaluation
        fused_results = {}
        for case_id, candidates in model_results.items():
            # Sort candidates by score and convert to expected format
            sorted_candidates = sorted(
                candidates.items(), key=lambda x: x[1], reverse=True
            )
            fused_results[case_id] = [
                {"candidate_id": candidate_id, "score": score}
                for candidate_id, score in sorted_candidates[:1000]
            ]
    else:
        # Set up fusion parameters
        distributions = {}

    # Load ground truth for evaluation (needed for grid search)
    try:
        if "get_data" in globals():
            _, _, relevant_docs = get_data(
                args.dataset_path, year=args.year, segment=args.data_split
            )
        else:
            relevant_docs = load_json(
                root / f"dataset/{args.data_split}_labels_{args.year}.json"
            )
            # Convert to set format expected by evaluation
            relevant_docs = {
                case_id: set(labels) for case_id, labels in relevant_docs.items()
            }
    except Exception:
        relevant_docs = {}

    # Continue with fusion logic
    if len(all_results) > 1:
        # Handle NSF grid search
        if args.fusion == "nsf" and args.nsf_grid_search:
            # Generate all weight combinations
            weight_combinations = generate_weight_combinations(
                num_models=len(all_results), resolution=args.weight_resolution
            )
            normalization_methods = ["min-max", "z-score", "percentile-rank"]

            # Store all grid search results
            grid_search_results = []
            system_names = list(all_results.keys())

            for norm_method in normalization_methods:
                for weight_combo in weight_combinations:
                    # Create weights dictionary
                    weights = {
                        system: weight
                        for system, weight in zip(system_names, weight_combo)
                    }

                    # Perform fusion
                    fused_results = Aggregator.fuse(
                        ranked_lists=all_results,
                        method="nsf",
                        normalization=norm_method,
                        linear_weights=weights,
                        percentile_distributions=distributions,
                    )

                    # Store results
                    result_entry = {
                        "normalization": norm_method,
                        "weights": weights,
                        "fused_results": fused_results,
                        "num_cases": len(fused_results),
                    }
                    grid_search_results.append(result_entry)

            # Save comprehensive grid search results
            models_identifier = "_".join(sorted(all_results.keys()))
            grid_output_file = f"fusion_scores_{args.fusion}_{models_identifier}_{args.year}_{args.data_split}.json"
            grid_output_path = Path(args.output_dir) / grid_output_file

            grid_summary = {
                "fusion_method": "nsf",
                "grid_search": True,
                "weight_resolution": args.weight_resolution,
                "systems_used": models_identifier,
                "total_experiments": len(grid_search_results),
                "fused_scores": grid_search_results,
            }

            with open(grid_output_path, "w") as f:
                json.dump(grid_summary, f, indent=2)

            return

        else:
            # Regular fusion (single configuration)
            weights = {}
            if args.fusion == "nsf":
                # Equal weights for all systems
                weights = {system: 1.0 / len(all_results) for system in all_results}

                normalization_methods = ["min-max", "z-score", "percentile-rank"]

                results = []
                for norm_method in normalization_methods:
                    # Perform fusion with this normalization
                    fused_results = Aggregator.fuse(
                        ranked_lists=all_results,
                        method=args.fusion,
                        normalization=norm_method,
                        linear_weights=weights,
                        percentile_distributions=distributions,
                    )

                    # Save results with normalization in filename
                    models_used = sorted(all_results.keys())
                    models_identifier = "_".join(models_used)
                    output_path = f"fusion_scores_{args.fusion}_{models_identifier}_{norm_method}_{args.year}_{args.data_split}.json"

                    
                    result_entry = {
                        "normalization": norm_method,
                        "weights": weights,
                        "fused_results": fused_results,
                        "num_cases": len(fused_results),
                    }
                    results.append(result_entry)

                grid_summary = {
                    "fusion_method": "nsf",
                    "grid_search": False,
                    "systems_used": models_identifier,
                    "total_experiments": len(results),
                    "fused_scores": results,
                }
                
                grid_output_file = f"fusion_scores_{args.fusion}_{models_identifier}_{args.year}_{args.data_split}.json"
                grid_output_path = Path(args.output_dir) / grid_output_file

                with open(grid_output_path, "w") as f:
                    json.dump(grid_summary, f, indent=2)

                return

            # Single normalization or non-NSF fusion
            fused_results = Aggregator.fuse(
                ranked_lists=all_results,
                method=args.fusion,
                normalization=args.normalization if args.fusion == "nsf" else None,
                linear_weights=weights if args.fusion == "nsf" else None,
                percentile_distributions=distributions,
            )

            models_used = sorted(all_results.keys())
            models_identifier = "_".join(models_used)
            # Include normalization in filename for NSF
            if (
                args.fusion == "nsf"
                and hasattr(args, "normalization")
                and args.normalization
            ):
                output_path = f"fusion_scores_{args.fusion}_{models_identifier}_{args.normalization}_{args.year}_{args.data_split}.json"
            else:
                output_path = f"fusion_scores_{args.fusion}_{models_identifier}_{args.year}_{args.data_split}.json"

            with open(Path(args.output_dir) / output_path, "w") as f:
                grid_summary = {
                    "fusion_method": args.fusion,
                    "grid_search": False,
                    "systems_used": models_used,
                    "total_experiments": 1,
                    "fused_scores": [fused_results],
                }
                json.dump(grid_summary, f, indent=2)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fusion analysis script for LCE framework"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path", type=str, default="dataset", help="Path to dataset"
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2025",
        choices=["2024", "2025"],
        help="Dataset year",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Data split to use",
    )

    # Model selection
    parser.add_argument("--run_bm25", action="store_true", help="Run BM25 baseline")
    parser.add_argument("--run_bert", action="store_true", help="Run BERT bi-encoder")
    parser.add_argument(
        "--run_xlm_roberta", action="store_true", help="Run XLM-RoBERTa bi-encoder"
    )
    parser.add_argument("--run_e5", action="store_true", help="Run E5 bi-encoder")
    parser.add_argument("--run_bge", action="store_true", help="Run BGE bi-encoder")
    parser.add_argument(
        "--run_monot5", action="store_true", help="Run MonoT5 cross-encoder"
    )
    parser.add_argument(
        "--run_monot5_3b", action="store_true", help="Run MonoT5 3B cross-encoder"
    )

    # Fusion arguments
    parser.add_argument(
        "--fusion",
        type=str,
        required=False,
        choices=["bcf", "rrf", "nsf"],
        help="Fusion method",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="min-max",
        choices=["min-max", "z-score", "percentile-rank"],
        help="Normalization method (for NSF)",
    )
    parser.add_argument(
        "--nsf_grid_search",
        action="store_true",
        help="Perform grid search for NSF fusion with all weight combinations",
    )
    parser.add_argument(
        "--weight_resolution",
        type=float,
        default=0.1,
        help="Resolution for weight grid search (default: 0.1)",
    )

    # Model training parameters
    parser.add_argument(
        "--dynamic_sampling_strategy",
        type=str,
        default="hard",
        help="Dynamic sampling strategy",
    )
    parser.add_argument(
        "--max_negatives_per_positive",
        type=int,
        default=3,
        help="Max negatives per positive",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="fusion_results", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
