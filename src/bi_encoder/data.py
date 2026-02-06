"""
Refactored BiEncoder Dataset Module

This module provides a clean, well-structured implementation of the BiEncoderDataset class
with improved organization, better error handling, and clearer separation of concerns.
"""

import logging
import os
import random
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from datasets import Dataset
from tqdm import tqdm

# Add parent directory to path before importing local modules
root = Path(os.path.realpath(__file__)).parents[2]  # Go up to framework root
sys.path.insert(0, str(root))

from utils.common import build_dataset, get_data
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types with their text formatting requirements."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"
    E5 = "e5"
    BGE_M3 = "bge_m3"


class AugmentationType(Enum):
    """Dataset augmentation strategies."""

    STANDARD = "standard"
    AUGMENTED = "augmented"


class SamplingStrategy(Enum):
    """Dynamic sampling strategies for negative selection."""

    RANDOM = "random"
    HARD = "hard"
    MIXED = "mixed"


@dataclass
class BiEncoderConfig:
    """Configuration for BiEncoder dataset creation."""

    data_path: str
    year: str
    segment: str = "train"
    architecture: str = "bi_encoder"
    augmentation_type: AugmentationType = AugmentationType.STANDARD
    max_negatives_per_positive: int = -1  # -1 means no limit
    dynamic_sampling: bool = False
    dynamic_sampling_strategy: SamplingStrategy = SamplingStrategy.RANDOM
    hard_negative_ratio: float = 0.5
    batch_size: int = 32
    random_seed: int = 42
    model_type: ModelType = ModelType.SENTENCE_TRANSFORMERS
    training_samples_file: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_negatives_per_positive == 0:
            raise ValueError(
                "max_negatives_per_positive must be -1 (unlimited) or positive integer"
            )
        if not 0 < self.hard_negative_ratio <= 1:
            raise ValueError("hard_negative_ratio must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class TextFormatter:
    """Handles text formatting for different model types."""

    @staticmethod
    def format_text(text: str, text_type: str, model_type: ModelType) -> str:
        """
        Format text according to model requirements.

        Args:
            text: Input text
            text_type: "query" or "passage"
            model_type: Type of model requiring specific formatting

        Returns:
            Formatted text string
        """
        if model_type == ModelType.E5:
            prefix = "query: " if text_type == "query" else "passage: "
            return prefix + text
        elif model_type == ModelType.BGE_M3:
            # BGE-M3 doesn't require prefixes but can handle longer sequences
            return text
        else:
            # Default sentence_transformers format
            return text


class NegativeSampler:
    """Handles various negative sampling strategies."""

    def __init__(self, config: BiEncoderConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

    def apply_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        """
        Apply the configured sampling strategy to negative candidates.

        Args:
            neg_candidates: List of negative candidate dictionaries
            case_id: Case ID (used for hard negative mining)

        Returns:
            Sampled negative candidates
        """
        if not neg_candidates:
            return neg_candidates

        if not self.config.dynamic_sampling:
            return self._apply_limit(neg_candidates)

        if self.config.dynamic_sampling_strategy == SamplingStrategy.RANDOM:
            return self._random_sampling(neg_candidates, num_pos_candidates)
        elif self.config.dynamic_sampling_strategy == SamplingStrategy.HARD:
            return self._hard_negative_sampling(neg_candidates, num_pos_candidates)
        elif self.config.dynamic_sampling_strategy == SamplingStrategy.MIXED:
            return self._mixed_sampling(neg_candidates, num_pos_candidates)
        else:
            return self._apply_limit(neg_candidates)

    def _apply_limit(self, candidates: List[Dict]) -> List[Dict]:
        """Apply negative limit with random sampling."""
        if self.config.max_negatives_per_positive <= 0:
            return candidates

        if len(candidates) > self.config.max_negatives_per_positive:
            return self.rng.sample(candidates, self.config.max_negatives_per_positive)
        return candidates

    def _random_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        """Apply random sampling strategy."""
        if self.config.max_negatives_per_positive <= 0:
            return neg_candidates

        desired_count = num_pos_candidates * self.config.max_negatives_per_positive
        sample_size = min(desired_count, len(neg_candidates))

        if sample_size <= 0:
            return []

        return self.rng.sample(neg_candidates, sample_size)

    def _hard_negative_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        """
        Apply hard negative sampling strategy.

        Note: This is a placeholder. In practice, you'd implement actual
        hard negative mining based on similarity scores or other criteria.
        """
        if self.config.max_negatives_per_positive <= 0:
            return neg_candidates

        # reduced_neg_candidates = neg_candidates[:20][::-1]

        desired_count = num_pos_candidates * self.config.max_negatives_per_positive
        end_index = min(desired_count, len(neg_candidates))
        return neg_candidates[:end_index]

    def _mixed_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        """
        Apply mixed sampling strategy combining random and hard negatives.
        """
        if self.config.max_negatives_per_positive <= 0:
            return neg_candidates

        desired_count = num_pos_candidates * self.config.max_negatives_per_positive
        sample_size = min(desired_count, len(neg_candidates))

        if sample_size <= 0:
            return []

        return self.rng.sample(neg_candidates, sample_size)


class DatasetStats:
    """Handles dataset statistics calculation and reporting."""

    @staticmethod
    def calculate_stats(data: Dataset, config: BiEncoderConfig) -> Dict:
        """Calculate comprehensive dataset statistics."""
        if not data:
            return DatasetStats._get_config_stats(config)

        total_positives = sum(1 for item in data if item["label"] == 1)
        total_negatives = sum(1 for item in data if item["label"] == 0)
        actual_ratio = total_negatives / total_positives if total_positives > 0 else 0

        stats = DatasetStats._get_config_stats(config)
        stats.update(
            {
                "total_positive_pairs": total_positives,
                "total_negative_pairs": total_negatives,
                "actual_negative_to_positive_ratio": round(actual_ratio, 2),
                "total_training_pairs": len(data),
            }
        )

        return stats

    @staticmethod
    def _get_config_stats(config: BiEncoderConfig) -> Dict:
        """Get configuration-based statistics."""
        return {
            "max_negatives_per_positive": config.max_negatives_per_positive,
            "dynamic_sampling": config.dynamic_sampling,
            "dynamic_sampling_strategy": (
                config.dynamic_sampling_strategy.value
                if config.dynamic_sampling
                else None
            ),
            "hard_negative_ratio": (
                config.hard_negative_ratio
                if config.dynamic_sampling_strategy == SamplingStrategy.MIXED
                else None
            ),
            "model_type": config.model_type.value,
            "augmentation_type": config.augmentation_type.value,
        }

    @staticmethod
    def print_stats(stats: Dict) -> None:
        """Print formatted dataset statistics."""
        print("=" * 60)
        print("BI-ENCODER DATASET CONFIGURATION & STATISTICS")
        print("=" * 60)
        print(f"Max negatives per positive: {stats['max_negatives_per_positive']}")
        print(f"Dynamic sampling enabled: {stats['dynamic_sampling']}")
        print(f"Model type: {stats['model_type']}")
        print(f"Augmentation type: {stats['augmentation_type']}")

        if stats["dynamic_sampling"]:
            print(f"Dynamic sampling strategy: {stats['dynamic_sampling_strategy']}")
            if stats["hard_negative_ratio"]:
                print(f"Hard negative ratio: {stats['hard_negative_ratio']}")

        if "total_positive_pairs" in stats:
            print("\nDataset Statistics:")
            print(f"Total positive pairs: {stats['total_positive_pairs']:,}")
            print(f"Total negative pairs: {stats['total_negative_pairs']:,}")
            print(f"Actual neg/pos ratio: {stats['actual_negative_to_positive_ratio']}")
            print(f"Total training pairs: {stats['total_training_pairs']:,}")

        print("=" * 60)


class BiEncoderDataset:
    """
    Refactored BiEncoder dataset class with improved organization and functionality.

    This class handles the creation of training datasets for bi-encoder models
    with support for various augmentation strategies, negative sampling, and
    model-specific text formatting.
    """

    def __init__(self, config: Union[BiEncoderConfig, dict], **kwargs):
        """
        Initialize the BiEncoder dataset.

        Args:
            config: Configuration object or dictionary
            **kwargs: Additional configuration parameters
        """
        # Handle both dict and BiEncoderConfig inputs
        if isinstance(config, dict):
            config.update(kwargs)
            self.config = BiEncoderConfig(**config)
        else:
            self.config = config

        print(self.config)
        # Initialize components
        self._setup_random_seeds()
        self.text_formatter = TextFormatter()
        self.negative_sampler = NegativeSampler(self.config)

        # Load and process data
        self._load_formatted_data(self.config)
        self.data = self._create_dataset()
        self.queries, self.corpus, self.relevant_docs = self._get_queries_corpus_docs()

        logger.info(
            f"BiEncoderDataset initialized with {len(self.data)} training pairs"
        )

    def _setup_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def _load_formatted_data(self, config: BiEncoderConfig) -> None:
        """Load and format raw data."""
        if (
            config.dynamic_sampling
            and config.dynamic_sampling_strategy == SamplingStrategy.RANDOM
        ) or config.segment == "dev":
            self.formatted_data = build_dataset(
                config.data_path,
                config.year,
                config.segment,
            )
        else:
            self.formatted_data = build_dataset(
                config.data_path,
                config.year,
                config.segment,
                config.training_samples_file,
            )

    def _create_dataset(self) -> Dataset:
        """Create HuggingFace Dataset based on augmentation type."""
        if self.config.augmentation_type == AugmentationType.STANDARD:
            return self._create_standard_dataset()
        else:
            # TODO: Implement other augmentation types
            logger.warning(
                f"Augmentation type {self.config.augmentation_type} not implemented, using standard"
            )
            return self._create_standard_dataset()

    def _create_standard_dataset(self) -> Dataset:
        """Create standard dataset without in-batch negatives."""
        training_data = {"sentence1": [], "sentence2": [], "label": []}

        for sample in tqdm(self.formatted_data, desc="Creating standard dataset"):
            self._process_sample_standard(sample, training_data)

        return Dataset.from_dict(training_data)

    def _process_sample_standard(self, sample: Dict, training_data: Dict) -> None:
        """Process a single sample for standard dataset creation."""
        for pos_cand in sample["pos_candidates"]:
            # Add positive pair
            self._add_pair(training_data, sample["text"], pos_cand["text"], label=1)

            # Process negatives
        neg_candidates = self.negative_sampler.apply_sampling(
            sample["neg_candidates"], len(sample["pos_candidates"])
        )

        # Add negative pairs
        for neg_cand in neg_candidates:
            self._add_pair(training_data, sample["text"], neg_cand["text"], label=0)

    def _add_pair(
        self, training_data: Dict, query: str, passage: str, label: int
    ) -> None:
        """Add a training pair to the dataset."""
        training_data["sentence1"].append(
            self.text_formatter.format_text(query, "query", self.config.model_type)
        )
        training_data["sentence2"].append(
            self.text_formatter.format_text(passage, "passage", self.config.model_type)
        )
        training_data["label"].append(label)

    def _get_queries_corpus_docs(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Set[str]]]:
        """Extract queries, corpus, and relevant documents."""
        queries = {}
        corpus = {}
        relevant_docs = {}

        for sample in self.formatted_data:
            queries[sample["id"]] = sample["text"]
            relevant_docs[sample["id"]] = set()

            for cand in sample["pos_candidates"]:
                corpus[len(corpus)] = cand["text"]
                relevant_docs[sample["id"]].add(len(corpus) - 1)

            for cand in sample["neg_candidates"]:
                corpus[len(corpus)] = cand["text"]

        return queries, corpus, relevant_docs

    def get_stats(self) -> Dict:
        """Get comprehensive dataset statistics."""
        return DatasetStats.calculate_stats(self.data, self.config)

    def print_stats(self) -> None:
        """Print dataset statistics."""
        stats = self.get_stats()
        DatasetStats.print_stats(stats)

    def __len__(self) -> int:
        """Return the number of training pairs."""
        return len(self.data) if self.data else 0

    def __getitem__(self, idx: int) -> Dict:
        """Get a training pair by index."""
        if not self.data:
            raise IndexError("Dataset is empty")
        return self.data[idx]


# Backwards compatibility functions
def create_bi_encoder_dataset(
    data_path: str, year: str, segment: str = "train", **kwargs
) -> BiEncoderDataset:
    """
    Factory function for creating BiEncoder datasets.

    Args:
        data_path: Path to the dataset
        year: Dataset year
        segment: Data segment
        **kwargs: Additional configuration parameters

    Returns:
        Configured BiEncoderDataset instance
    """
    config = BiEncoderConfig(data_path=data_path, year=year, segment=segment, **kwargs)
    return BiEncoderDataset(config)
