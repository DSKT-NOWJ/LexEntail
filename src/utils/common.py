"""
Utility functions for COLIEE Task 2 data processing.

This module provides various utility functions for data loading, preprocessing,
text processing, and dataset manipulation.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import spacy


PUNCTUATION = ".,!?"

# Initialize spaCy
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def build_dataset(dataset_path, year="2025", segment="train", training_samples_file=None):
    corpus_dir, cases_dir, label_data = get_data(dataset_path, year=year, segment=segment)

    print(f"Building {segment} dataset for year: {year}")
    cnt = 0
    training_samples = {}
    if training_samples_file:
        training_samples = load_json(training_samples_file)
    
    dataset = []
    for case in cases_dir:
        base_case_file = corpus_dir / case / "entailed_fragment.txt"
        base_case_data = preprocess_case_data(base_case_file, uncased=False)
        label = label_data[case]

        case_dict = {
            "id": case,
            "text": base_case_data,
            "pos_candidates": [],
            "neg_candidates": [],
        }

        candidate_dir = corpus_dir / case / "paragraphs"
        if not training_samples_file:
            candidate_cases = sorted(os.listdir(candidate_dir))
            cnt += len(candidate_cases)
        else:
            candidate_cases = training_samples[case]
            cnt += len(candidate_cases)

        for cand_case in candidate_cases:
            # if case in training_samples and cand_case not in training_samples[case]:
            #     continue
            cand_case_file = candidate_dir / cand_case
            cand_case_data = preprocess_case_data(
                cand_case_file, uncased=False, filter_min_length=10
            )
            
            if cand_case_data is None:
                continue
            l = "pos_candidates" if cand_case in label else "neg_candidates"
            case_dict[l].append({"id": cand_case, "text": cand_case_data})
        dataset.append(case_dict)
        
    print(f"{segment}: {cnt} cases processed")
    return dataset


def get_data(data_path, year, segment="train"):
    """
    Get data paths and labels for a specific segment and year.
    
    Args:
        data_path: Path to the dataset
        year: Dataset year ('2024' or '2025')
        segment: Data segment ('train', 'dev', or 'test')
        
    Returns:
        tuple: (corpus_dir, cases_dir, label_data)
        
    Raises:
        ValueError: If invalid year is provided
    """
    if year == '2024':
        if segment == 'train':
            start_idx, end_idx = 0, 625
        elif segment == 'dev':
            start_idx, end_idx = 625, 725
        elif segment == 'test':
            start_idx, end_idx = 725, 825
    elif year == '2025':
        if segment == 'train':
            start_idx, end_idx = 0, 725
        elif segment == 'dev':
            start_idx, end_idx = 725, 825
        elif segment == 'test':
            start_idx, end_idx = 825, 925
    else:
        raise ValueError(f"Invalid year: {year}")

    root_dir = Path(data_path)
    corpus_dir = root_dir / f"task2_train_files_{year}"
    cases_dir = sorted(os.listdir(corpus_dir))

    # Load appropriate labels based on segment
    if segment == "train":
        label_data = load_json(root_dir / f"train_labels_{year}.json")
    elif segment == "dev":
        label_data = load_json(root_dir / f"dev_labels_{year}.json")
    else:  # test
        label_data = load_json(root_dir / f"test_labels_{year}.json")

    return corpus_dir, cases_dir[start_idx:end_idx], label_data


def save_txt(file_path, text):
    """
    Save text to a file.
    
    Args:
        file_path: Path to save the file
        text: Text content to save
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def load_txt(file_path, skip=0):
    """
    Load text from a file.
    
    Args:
        file_path: Path to the file
        skip: Number of lines to skip from the beginning
        
    Returns:
        str: File content
    """
    with open(file_path, encoding="utf-8") as f:
        while skip > 0:
            f.readline()
            skip -= 1
        data = f.read()
    return data


def load_json(file_path):
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    """
    Save data to a JSON file.
    
    Args:
        file_path: Path to save the JSON file
        data: Data to save
    """
    with open(file_path, "w+", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_sentences(doc):
    """
    Split document into sentences using spaCy.
    
    Args:
        doc: Input document text
        
    Returns:
        list: List of sentences
    """
    doc = nlp(doc)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def filter_document(doc, min_sentence_length=None):
    """
    Filter document by removing short sentences.
    
    Args:
        doc: Input document text
        min_sentence_length: Minimum sentence length in words
        
    Returns:
        str: Filtered document
    """
    sentences = get_sentences(doc)
    if min_sentence_length:
        sentences = [
            sent for sent in sentences 
            if len(sent.split()) >= min_sentence_length
        ]
    doc = " ".join(sentences)
    return doc


def handle_base_case(content: str) -> list[str]:
    """
    Parse base case content into sections using regex pattern.
    
    Args:
        content: Raw base case content
        
    Returns:
        list: List of parsed sections
    """
    pattern = r'(?ms)(^\s*\[\d+\]\s*.*?)(?=^\s*\[\d+\]|\Z)'
    sections = re.findall(pattern, content)
    sections = [section.strip() for section in sections]
    return sections


def process_dataset(data_path, labels_path):
    """
    Process the complete dataset with labels.
    
    Args:
        data_path: Path to dataset directory
        labels_path: Path to labels JSON file
        
    Returns:
        list: Processed dataset with cases, queries, and labels
    """
    data = []
    labels = {}
    
    # Load labels
    with open(labels_path, 'r') as f:
        content = json.load(f)
        for case, label in content.items():
            labels[case] = [x.split('.')[0] for x in label]
    
    # Process each case
    for case in os.listdir(data_path):
        case_path = os.path.join(data_path, case)
        
        # Load paragraphs
        paragraphs = []
        paragraphs_dir = os.path.join(case_path, 'paragraphs')
        for paragraph in os.listdir(paragraphs_dir):
            with open(os.path.join(paragraphs_dir, paragraph), 'r') as f:
                paragraphs.append(f.read())
                
        # Load base case
        with open(os.path.join(case_path, 'base_case.txt'), 'r') as f:
            base_case = f.read()
            
        # Load query (entailed fragment)
        with open(os.path.join(case_path, 'entailed_fragment.txt'), 'r') as f:
            query = f.read()
            
        data.append({
            'id': case,
            'query': query,
            'base_case': handle_base_case(base_case),
            'paragraphs': paragraphs,
            'labels': labels[case]
        })
            
    return data


def segment_document(doc, max_sent_per_segment, stride, max_segment_len=None):
    """
    Segment document into overlapping chunks.
    
    Args:
        doc: Input document text
        max_sent_per_segment: Maximum sentences per segment
        stride: Step size between segments
        max_segment_len: Maximum segment length in words
        
    Returns:
        list: List of document segments
    """
    sentences = get_sentences(doc)
    segments = []
    
    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i:i + max_sent_per_segment])
        
        if max_segment_len:
            segment = " ".join(segment.split()[:max_segment_len])
        segments.append(segment)
        
    return segments


def preprocess_case_data(file_path, max_length=None, min_sentence_length=None,
                        uncased=False, filter_min_length=None):
    """
    Preprocess case data from file with various filtering options.
    
    Args:
        file_path: Path to the case file
        max_length: Maximum length in words
        min_sentence_length: Minimum sentence length for filtering
        uncased: Whether to convert to lowercase
        filter_min_length: Minimum document length in words
        
    Returns:
        str or None: Preprocessed text or None if file doesn't exist or is too short
    """
    if not os.path.exists(file_path):
        return None

    text = load_txt(file_path)

    # Clean and normalize text
    text = (
        text.strip()
        .replace("\n", " ")
        .replace("FRAGMENT_SUPPRESSED", "")
        .replace("FACTUAL", "")
        .replace("BACKGROUND", "")
        .replace("ORDER", "")
    )
    
    if uncased:
        text = text.lower()
        
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w])

    # Handle citation numbers
    cite_number = re.search(r"\[[0-9]+\]", text)
    if cite_number:
        start, end = cite_number.span()
        text = text[:start].strip() + ' ' + text[end:].strip()
        
    # Apply filters
    if filter_min_length:
        words = text.split()
        if len(words) <= filter_min_length:
            return None

    if min_sentence_length:
        text = filter_document(text, min_sentence_length)
        
    if max_length:
        words = text.split()[:max_length]
        text = " ".join(words)
        
    # Ensure text ends with period
    if not text.endswith("."):
        text = text + "."
        
    return text


def format_output(text):
    """
    Format output by removing HTML tags and converting to lowercase.
    
    Args:
        text: Input text
        
    Returns:
        str: Formatted text
    """
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", text)
    return cleantext.strip().lower()


def format_output_2(text):
    """
    Extract document numbers from text using regex.
    
    Args:
        text: Input text containing document references
        
    Returns:
        list: List of document indices (0-based)
    """
    regex = r"Document (\d+)"
    numbers = re.findall(regex, text)

    if numbers:
        return [int(num) - 1 for num in numbers]

    print(f"Parsing error: No valid match found in '{text}'")
    return []


def train_test_split(data, test_size: float, random_state: int) -> tuple:
    """
    Split data into train and test sets.
    
    Args:
        data: Input data list
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_data, test_data)
    """
    np.random.seed(random_state)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    split_idx = int(len(data) * test_size)
    
    train_data = [data[i] for i in indices[split_idx:]]
    test_data = [data[i] for i in indices[:split_idx]]
    
    return train_data, test_data


def load_dataset(path):
    """
    Load dataset from JSON file.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        dict: Loaded dataset
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)