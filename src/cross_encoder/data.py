import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Add parent directory to path before importing local modules
root = Path(os.path.realpath(__file__)).parents[2]  # Go up to framework root
sys.path.insert(0, str(root))
from utils.common import build_dataset


class CrossEncoderDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        year: str,
        segment: str = "train",
        architecture: str = "cross_encoder",
        max_neg_per_pos: int = -1,
        ns_strategy: str = "hard",
        training_samples_file: str | None = None,
    ) -> None:
        self.data_path = data_path
        self.year = year
        self.segment = segment
        self.architecture = architecture
        self.max_neg_per_pos = max_neg_per_pos
        self.ns_strategy = ns_strategy
        self.training_samples_file = training_samples_file
        self.formatted_data = self._load_formatted_data()
        self.rng = random.Random(42)

        self.data = self.create_dataset(self.formatted_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Union[Dict, List[Tuple]]:
        return self.data[index]
    
    def _load_formatted_data(self) -> None:
        """Load and format raw data."""
        if self.ns_strategy == "random" or self.segment == "dev":
            formatted_data = build_dataset(
                self.data_path,
                self.year,
                self.segment,
            )
        else:
            formatted_data = build_dataset(
                self.data_path,
                self.year,
                self.segment,
                self.training_samples_file,
            )
        return formatted_data

    def create_dataset(self, formatted_data: List[Dict]) -> List[Tuple[str, str, int]]:
        """Create dataset for cross-encoder: individual (query, doc, label) tuples."""
        training_data = []
        for sample in formatted_data:
            # Add positive pairs
            batch = []
            for cand in sample["pos_candidates"]:
                batch.append((sample["text"], cand["text"], 1))

            # Add negative pairs (limit to avoid class imbalance)
            # if self.max_neg_per_pos < 0:
            #     max_negatives = len(sample["neg_candidates"])
            # else:
            #     max_negatives = min(
            #         len(sample["neg_candidates"]),
            #         self.max_neg_per_pos * len(sample["pos_candidates"]),
            #     )
            neg_candidates = self._apply_sampling(
                sample["neg_candidates"], len(sample["pos_candidates"])
            )
            for cand in neg_candidates:
                batch.append((sample["text"], cand["text"], 0))
            training_data.append(batch)

        return training_data

    def _apply_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        if not neg_candidates:
            return neg_candidates

        if self.ns_strategy == "hard":
            return self._hard_negative_sampling(neg_candidates, num_pos_candidates)
        elif self.ns_strategy == "random":
            return self._random_negative_sampling(neg_candidates, num_pos_candidates)
        else:
            raise ValueError(f"Invalid negative sampling strategy: {self.ns_strategy}")

    def _hard_negative_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        if self.max_neg_per_pos < 0:
            return neg_candidates

        return neg_candidates[
            : min(num_pos_candidates * self.max_neg_per_pos, len(neg_candidates))
        ]

    def _random_negative_sampling(
        self, neg_candidates: List[Dict], num_pos_candidates: int
    ) -> List[Dict]:
        if self.max_neg_per_pos < 0:
            return neg_candidates

        return self.rng.sample(
            neg_candidates,
            min(num_pos_candidates * self.max_neg_per_pos, len(neg_candidates)),
        )


class GwenBatchCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, device: torch.device, max_length: int = 512
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.instruction = "Given a legal statement and a document, determine if the document is relevant to the statement"

    def __call__(self, batch: List[Tuple[str, str, int]]) -> Dict[str, torch.Tensor]:
        """Collate for cross-encoder: concatenated query+doc pairs."""
        # For causal LM reranking, we format as a prompt that ends with "Relevant:"
        # and expect the model to generate "yes" or "no" as the next token
        texts = [
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}\nIs the document relevant to the query?:".format(
                instruction=self.instruction, query=query, doc=doc
            )
            for b in batch
            for query, doc, _ in b
        ]

        # Create the targets (what should come after "Relevant:")
        targets = [
            " yes" if example[2] == 1 else " no"  # Include space before token
            for b in batch
            for example in b
        ]

        # Tokenize inputs and targets separately (don't return tensors yet)
        input_encodings = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_tensors=None,  # Return lists, not tensors
            max_length=self.max_length - 10,  # Leave room for target tokens
        )

        target_encodings = self.tokenizer(
            targets,
            padding=False,
            add_special_tokens=False,  # Don't add BOS/EOS to targets
            return_tensors=None,  # Return lists, not tensors
        )

        # Combine input and target for causal LM training
        combined_input_ids = []
        combined_attention_mask = []
        labels_list = []

        for i in range(len(texts)):
            input_ids = torch.tensor(input_encodings["input_ids"][i])
            target_ids = torch.tensor(target_encodings["input_ids"][i])

            # Combine input and target
            full_sequence = torch.cat([input_ids, target_ids], dim=0)
            combined_input_ids.append(full_sequence)

            # Create attention mask
            attention_mask = torch.ones_like(full_sequence)
            combined_attention_mask.append(attention_mask)

            # Create labels: -100 for input tokens, actual token IDs for target tokens
            labels = torch.full_like(full_sequence, -100)
            labels[len(input_ids) :] = target_ids  # Only compute loss on target tokens
            labels_list.append(labels)

        # Pad sequences to same length
        max_len = max(len(seq) for seq in combined_input_ids)
        max_len = min(max_len, self.max_length)  # Respect max_length

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for i in range(len(combined_input_ids)):
            seq_len = len(combined_input_ids[i])
            if seq_len > max_len:
                # Truncate if too long
                padded_input_ids.append(combined_input_ids[i][:max_len])
                padded_attention_mask.append(combined_attention_mask[i][:max_len])
                padded_labels.append(labels_list[i][:max_len])
            else:
                # Pad if too short
                pad_length = max_len - seq_len
                padded_input_ids.append(
                    torch.cat(
                        [
                            combined_input_ids[i],
                            torch.full((pad_length,), self.tokenizer.pad_token_id),
                        ]
                    )
                )
                padded_attention_mask.append(
                    torch.cat([combined_attention_mask[i], torch.zeros(pad_length)])
                )
                padded_labels.append(
                    torch.cat([labels_list[i], torch.full((pad_length,), -100)])
                )

        tokenized = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

        for key in tokenized:
            if isinstance(tokenized[key], torch.Tensor):
                tokenized[key] = tokenized[key].to(self.device)

        return tokenized


class MonoT5BatchCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, device: torch.device, max_length: int = 512
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.pattern = "Query: {} Document: {} Relevant:"

    def __call__(self, batch: List[Tuple[str, str, int]]) -> Dict[str, torch.Tensor]:
        """Collate for cross-encoder: concatenated query+doc pairs."""
        texts = [
            self.pattern.format(example[0], example[1]) for b in batch for example in b
        ]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length,
        )
        tokenized["labels"] = self.tokenizer(
            ["true" if example[2] == 1 else "false" for b in batch for example in b],
            return_tensors="pt",
        )["input_ids"]

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)
        return tokenized
