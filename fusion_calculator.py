#!/usr/bin/env python3
"""
Interactive Fusion Score Calculator

Input models, fusion techniques, and transformations to get fusion scores.
Returns the actual fusion scores, not evaluation metrics.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class FusionCalculator:
    """Calculate fusion scores for multiple models using different techniques."""

    def __init__(self):
        self.available_models = [
            "mbert", "roberta_large", "e5_large", "bge_m3",
            "monot5_base", "monot5_3b", "bm25"
        ]
        self.available_fusion_methods = ["bcf", "rrf", "nsf"]
        self.available_transformations = ["min-max", "z-score", "percentile-rank"]

    def load_model_scores(self, model_name: str, year: str = "2025") -> Dict:
        """Load scores for a specific model."""
        score_files = [
            f"scores_{model_name}_{year}_hard_neg3.json",
            f"scores_{model_name}_{year}_hard_neg5.json",
            f"scores_{model_name}_{year}_random_neg3.json",
            f"scores_{model_name}_{year}_random_neg5.json",
            f"scores_{model_name}_{year}_top20_neg3.json",
            f"scores_{model_name}_{year}_top20_neg5.json",
        ]

        # Try to find existing score file
        for score_file in score_files:
            if os.path.exists(score_file):
                with open(score_file, "r") as f:
                    return json.load(f)

        print(f"Warning: No score file found for {model_name}")
        return {}

    def transform_scores(
        self,
        results: Dict[str, float],
        transformation: str,
        percentile_distr: np.array = None,
    ) -> Dict[str, float]:
        """Transform scores using various normalization methods."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if transformation == "borda-count":
            num_candidates = len(results)
            return {
                pid: (num_candidates - idx + 1) / num_candidates
                for idx, pid in enumerate(results.keys())
            }

        elif transformation == "reciprocal-rank":
            return {pid: 1 / (60 + idx + 1) for idx, pid in enumerate(results.keys())}

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

    def weight_scores(self, results: Dict[str, float], w: float) -> Dict[str, float]:
        """Weight the scores by a constant factor."""
        return {candidate_id: score * w for candidate_id, score in results.items()}

    def aggregate_scores(self, *args: Dict[str, float]) -> Dict[str, float]:
        """Aggregate scores from multiple systems."""
        agg_results = defaultdict(float)
        for results in args:
            if results:
                for candidate_id, score in results.items():
                    agg_results[candidate_id] += score
        return dict(agg_results)

    def calculate_fusion_scores(
        self,
        models: List[str],
        fusion_method: str,
        transformation: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        year: str = "2025",
        case_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate fusion scores for given models and parameters.

        Args:
            models: List of model names to fuse
            fusion_method: 'bcf', 'rrf', or 'nsf'
            transformation: For NSF - 'min-max', 'z-score', 'percentile-rank'
            weights: Optional weights for models (for NSF)
            year: Dataset year
            case_ids: Optional list of specific case IDs to process

        Returns:
            Dict[case_id, Dict[candidate_id, score]]
        """
        print(f"Loading scores for models: {models}")

        # Load model scores
        all_results = {}
        for model in models:
            scores = self.load_model_scores(model, year)
            if scores:
                all_results[model] = scores
                print(f"✓ Loaded {model}")
            else:
                print(f"✗ Failed to load {model}")

        if not all_results:
            print("Error: No model scores loaded")
            return {}

        print(f"Using fusion method: {fusion_method}")
        if fusion_method == "nsf" and transformation:
            print(f"Using transformation: {transformation}")

        # Get case IDs to process
        if case_ids is None:
            case_ids = list(next(iter(all_results.values())).keys())

        # Set default weights for NSF
        if fusion_method == "nsf" and weights is None:
            weights = {model: 1.0 / len(all_results) for model in all_results}
            print(f"Using equal weights: {weights}")
        elif fusion_method == "nsf" and isinstance(weights, list):
            # Convert list weights to dictionary mapping model names to weights
            model_names = list(all_results.keys())
            if len(weights) != len(model_names):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_names)})")
            weights = {model: weight for model, weight in zip(model_names, weights)}
            print(f"Using provided weights: {weights}")

        # Calculate fusion scores
        fusion_results = {}

        for case_id in case_ids:
            case_fusion_results = []

            for system, system_results in all_results.items():
                case_scores = system_results.get(case_id, {})

                if not case_scores:
                    continue

                if fusion_method == "bcf":
                    case_scores = self.transform_scores(
                        case_scores, transformation="borda-count"
                    )
                elif fusion_method == "rrf":
                    case_scores = self.transform_scores(
                        case_scores, transformation="reciprocal-rank"
                    )
                elif fusion_method == "nsf":
                    case_scores = self.transform_scores(
                        case_scores, transformation=transformation
                    )
                    if weights:
                        case_scores = self.weight_scores(
                            case_scores, w=weights[system]
                        )

                case_fusion_results.append(case_scores)

            # Aggregate scores for this case
            fusion_results[case_id] = self.aggregate_scores(*case_fusion_results)

        print(f"✓ Fusion completed for {len(fusion_results)} cases")
        return fusion_results


def main():
    """Interactive main function."""
    calculator = FusionCalculator()

    print("=== Fusion Score Calculator ===")
    print("Available models:", calculator.available_models)
    print("Available fusion methods:", calculator.available_fusion_methods)
    print("Available transformations:", calculator.available_transformations)

    # # Example usage
    # print("\n=== Example Usage ===")

    # # Example 1: BCF fusion
    # models = ["monot5_base", "monot5_3b"]
    # fusion_scores = calculator.calculate_fusion_scores(
    #     models=models,
    #     fusion_method="bcf",
    #     year="2025"
    # )

    # print(f"\nExample BCF fusion results for {len(fusion_scores)} cases:")
    # if fusion_scores:
    #     case_id = list(fusion_scores.keys())[0]
    #     print(f"Case {case_id} top 5 candidates:")
    #     sorted_candidates = sorted(
    #         fusion_scores[case_id].items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )[:5]
    #     for candidate_id, score in sorted_candidates:
    #         print(f"  {candidate_id}: {score:.4f}")

    # # Example 2: NSF fusion with z-score
    # print(f"\n=== NSF with Z-Score Example ===")
    # fusion_scores_nsf = calculator.calculate_fusion_scores(
    #     models=["mbert", "roberta_large"],
    #     fusion_method="nsf",
    #     transformation="z-score",
    #     year="2025"
    # )

    # if fusion_scores_nsf:
    #     case_id = list(fusion_scores_nsf.keys())[0]
    #     print(f"Case {case_id} top 5 candidates:")
    #     sorted_candidates = sorted(
    #         fusion_scores_nsf[case_id].items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )[:5]
    #     for candidate_id, score in sorted_candidates:
    #         print(f"  {candidate_id}: {score:.4f}")

    return calculator


if __name__ == "__main__":
    calculator = main()
    
    # models = ["monot5-base", "monot5-3b"]
    models=['bert-base-multilingual-cased', "xlm-roberta-large", 'multilingual-e5-large', 'bge-m3', 'monot5-base', 'monot5-3b']
    scores = calculator.calculate_fusion_scores(
        models=models,
        fusion_method="nsf",
        transformation="z-score",
        weights=[0.0, 0.0, 0.1, 0.0, 0.1, 0.8],
        year="2024"
    )
    
    with open(f"2024_fusion_scores_{'_'.join(models)}_recall@20.json", 'w') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2, default=str)

#     print("\n=== Ready for Custom Usage ===")
#     print("Use calculator.calculate_fusion_scores() with your parameters:")
#     print("""
# # Example calls:
# scores = calculator.calculate_fusion_scores(
#     models=["mbert", "roberta_large", "e5_large"],
#     fusion_method="nsf",
#     transformation="min-max",
#     year="2025"
# )

# scores = calculator.calculate_fusion_scores(
#     models=["bge_m3", "monot5_3b"],
#     fusion_method="rrf",
#     year="2024"
# )

# scores = calculator.calculate_fusion_scores(
#     models=["mbert", "roberta_large"],
#     fusion_method="bcf",
#     case_ids=["case_001", "case_002"]  # Process specific cases only
# )
# """)