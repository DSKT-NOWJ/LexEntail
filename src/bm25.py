"""
BM25 search implementation for COLIEE Task 2.

This module provides functionality for creating BM25 indexes, performing searches,
and extracting negative samples for training data.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import jsonlines
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Set up root path
root = Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(root))

# Import after path setup
from utils.common import (
    get_data,
    load_json,
    preprocess_case_data,
    save_json,
    segment_document,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BM25 processing for COLIEE Task 2")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Data directory path"
    )
    parser.add_argument(
        "--year", type=str, default="2025", help="Dataset year (2024 or 2025)"
    )
    parser.add_argument(
        "--num_negative",
        type=int,
        default=10,
        help="Number of negative samples to extract",
    )
    parser.add_argument(
        "--training_samples_file",
        type=str,
        default=None,
        help="Path to training samples file",
    )
    return parser.parse_args()


def predict_bm25(searcher, doc, case):
    """
    Predict BM25 scores for a given document and case.

    Args:
        searcher: LuceneSearcher instance
        doc: Document text to search with
        case: Case identifier

    Returns:
        dict: BM25 scores for each candidate
    """
    bm25_score = defaultdict(lambda: 0)
    hits = []
    segments = segment_document(doc, 1, 1)

    for segment in segments:
        _hits = searcher.search(segment[:1024], k=100000)
        hits.extend(_hits)

    for hit in hits:
        if hit.docid.endswith("task2"):
            if hit.docid.split("_candidate")[0] == case:
                hit.docid = hit.docid.split("_task2")[0].split("_candidate")[1]
                bm25_score[hit.docid] = max(hit.score, bm25_score[hit.docid])

    return bm25_score


def predict_all_bm25(
    dataset_path, year, bm25_index_path, eval_segment="test", k1=None, b=None, topk=None
):
    """
    Predict BM25 scores for all cases in the dataset.

    Args:
        dataset_path: Path to dataset
        year: Dataset year
        bm25_index_path: Path to BM25 index
        eval_segment: Evaluation segment (train/dev/test)
        k1: BM25 k1 parameter
        b: BM25 b parameter
        topk: Number of top candidates to return

    Returns:
        dict: BM25 scores for all cases
    """
    searcher = LuceneSearcher(bm25_index_path)

    if k1 and b:
        print(f"BM25 parameters - k1: {k1}, b: {b}")
        searcher.set_bm25(k1, b)

    corpus_dir, cases_dir, _ = get_data(dataset_path, year=year, segment=eval_segment)
    bm25_scores = {}

    for case in cases_dir:
        base_case_data = preprocess_case_data(
            corpus_dir / case / "entailed_fragment.txt"
        )
        score = predict_bm25(searcher, base_case_data, case)

        if topk is not None:
            sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)[
                :topk
            ]
            score = {x[0]: x[1] for x in sorted_score}

        bm25_scores[case] = score

    return bm25_scores


def create_bm25_indexes(args):
    """
    Create BM25 indexes for all segments (train, dev, test).

    Args:
        args: Parsed command line arguments
    """
    tmp_dir = root / "dataset/bm25_indexes/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for segment in ["train", "dev", "test"]:
        print(f"Creating BM25 index for {segment} segment...")
        indexes_dir = root / f"dataset/bm25_indexes/coliee_task2/{args.year}/{segment}"
        os.makedirs(indexes_dir, exist_ok=True)

        corpus_dir, cases_dir, _ = get_data(
            dataset_path, year=args.year, segment=segment
        )

        # Clear temporary file
        tmp_file = f"{tmp_dir}/candidate_{args.year}.jsonl"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        for case in tqdm(cases_dir, desc=f"Processing {segment} cases"):
            candidate_dir = corpus_dir / case / "paragraphs"
            candidate_cases = sorted(os.listdir(candidate_dir))

            for cand_case in candidate_cases:
                cand_case_file = candidate_dir / cand_case
                cand_case_data = preprocess_case_data(cand_case_file)
                cand_num = cand_case.split(".txt")[0]

                dict_ = {
                    "id": f"{case}_candidate{cand_num}.txt_task2",
                    "contents": cand_case_data,
                }

                with jsonlines.open(tmp_file, mode="a") as writer:
                    writer.write(dict_)

        # Build Lucene index
        subprocess.run(
            [
                "python",
                "-m",
                "pyserini.index.lucene",
                "-collection",
                "JsonCollection",
                "-generator",
                "DefaultLuceneDocumentGenerator",
                "-threads",
                "1",
                "-input",
                str(tmp_dir),
                "-index",
                str(indexes_dir),
                "-storePositions",
                "-storeDocvectors",
                "-storeRaw",
            ]
        )


def extract_negative_samples(args, segment="train"):
    """
    Extract negative samples using BM25 scores.

    Args:
        args: Parsed command line arguments
        segment: Data segment to process
    """
    print(f"Extracting negative samples for {segment} segment...")
    bm25_index_path = str(
        root / f"dataset/bm25_indexes/coliee_task2/{args.year}/{segment}"
    )

    _, cases_dir, label_data = get_data(dataset_path, year=args.year, segment=segment)
    bm25_scores = predict_all_bm25(
        dataset_path,
        year=args.year,
        bm25_index_path=bm25_index_path,
        eval_segment=segment,
        k1=2.5,
        b=0.2,
        topk=1000,
    )

    num_negatives = args.num_negative
    sample_dict = {}

    for case in tqdm(cases_dir, desc="Extracting negatives"):
        bm25_score = bm25_scores[case]
        top_negatives = sorted(bm25_score.items(), key=lambda x: x[1], reverse=True)[
            :num_negatives
        ]
        negative_ids = [x[0] for x in top_negatives]
        sample_dict[case] = list(set(negative_ids + label_data[case]))

    save_path = root / f"dataset/task2_{segment}_negatives_{args.year}.json"
    save_json(save_path, sample_dict)
    print(f"Negative samples saved to {save_path}")


def split_dataset(args):
    """
    Split dataset into train/dev/test based on year and predefined ranges.

    Args:
        args: Parsed command line arguments
    """
    print(f"Splitting dataset for year {args.year}...")
    label_data = load_json(root / f"dataset/task2_train_labels_{args.year}.json")

    if args.year == "2024":
        train_labels = {k: v for k, v in label_data.items() if int(k) in range(626)}
        dev_labels = {k: v for k, v in label_data.items() if int(k) in range(626, 726)}
        test_labels = {k: v for k, v in label_data.items() if int(k) in range(726, 826)}
    else:  # 2025
        train_labels = {k: v for k, v in label_data.items() if int(k) in range(726)}
        dev_labels = {k: v for k, v in label_data.items() if int(k) in range(726, 826)}
        test_labels = {k: v for k, v in label_data.items() if int(k) in range(826, 926)}

    # Save split datasets
    save_json(root / f"dataset/train_labels_{args.year}.json", train_labels)
    save_json(root / f"dataset/dev_labels_{args.year}.json", dev_labels)
    save_json(root / f"dataset/test_labels_{args.year}.json", test_labels)

    print(f"Dataset split complete:")
    print(f"  Train: {len(train_labels)} cases")
    print(f"  Dev: {len(dev_labels)} cases")
    print(f"  Test: {len(test_labels)} cases")


def main():
    """Main execution function."""
    args = parse_args()
    global dataset_path
    dataset_path = root / "dataset"

    print("Starting BM25 processing pipeline...")
    print(f"Dataset path: {dataset_path}")
    print(f"Year: {args.year}")

    # Clean up existing indexes
    shutil.rmtree(
        root / f"dataset/bm25_indexes/coliee_task2/{args.year}", ignore_errors=True
    )

    # Execute pipeline
    split_dataset(args)
    create_bm25_indexes(args)
    extract_negative_samples(args, segment="train")

    print("BM25 processing pipeline completed successfully!")


if __name__ == "__main__":
    main()
