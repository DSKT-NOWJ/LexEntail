# COLIEE Task 2 Framework

A framework for legal case entailment detection as part of the COLIEE (Competition on Legal Information Extraction/Entailment) Task 2 challenge. The framework supports bi-encoder and cross-encoder architectures with YAML-based configuration management.

## Project Structure

```
framework/
├── configs/
│   ├── bi_encoder.yaml       # Bi-encoder model configurations
│   └── cross_encoder.yaml    # Cross-encoder model configurations
├── scripts/
│   ├── train.sh              # Train all models (bi + cross)
│   ├── train_bi.sh           # Train a single bi-encoder
│   ├── train_cross.sh        # Train a single cross-encoder
│   ├── infer_bi.sh           # Run inference for bi-encoders
│   ├── infer_cross.sh        # Run inference for a cross-encoder
│   └── analyze.sh            # Run fusion analysis
├── src/
│   ├── train.py              # Training entry point
│   ├── infer_bi.py           # Bi-encoder inference
│   ├── infer_cross.py        # Cross-encoder inference
│   └── analyze.py            # Fusion analysis
├── dataset/                  # COLIEE data directory
└── checkpoints/              # Saved model checkpoints
```

## Supported Models

### Bi-Encoders
| Model | HuggingFace ID |
|-------|----------------|
| BERT Multilingual | `google-bert/bert-base-multilingual-cased` |
| XLM-RoBERTa Large | `FacebookAI/xlm-roberta-large` |
| Multilingual E5 | `intfloat/multilingual-e5-large` |
| BGE-M3 | `BAAI/bge-m3` |

### Cross-Encoders
| Model | HuggingFace ID |
|-------|----------------|
| MonoT5 Base | `castorini/monot5-base-msmarco` |
| MonoT5 3B | `castorini/monot5-3b-msmarco` |

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Preparation

Place COLIEE Task 2 data in the following structure:
```
dataset/
├── task2_train_files_2024/
├── task2_train_files_2025/
├── task2_train_labels_2024.json
├── task2_train_negatives_2025.json
└── dev_labels_2025.json
```

## Scripts

### Training

#### Train All Models
Trains all models defined in both `configs/bi_encoder.yaml` and `configs/cross_encoder.yaml`:

```bash
./scripts/train.sh
```

Options:
- `--bi-only` - Train only bi-encoders
- `--cross-only` - Train only cross-encoders
- `--skip-confirmation` - Skip confirmation prompts

#### Train a Single Bi-Encoder
```bash
./scripts/train_bi.sh "google-bert/bert-base-multilingual-cased"
./scripts/train_bi.sh "FacebookAI/xlm-roberta-large"
./scripts/train_bi.sh "intfloat/multilingual-e5-large"
./scripts/train_bi.sh "BAAI/bge-m3"
```

#### Train a Single Cross-Encoder
```bash
./scripts/train_cross.sh "castorini/monot5-base-msmarco"
./scripts/train_cross.sh "castorini/monot5-3b-msmarco"
```

### Inference

#### Bi-Encoder Inference
Runs inference for all bi-encoder models:

```bash
./scripts/infer_bi.sh
```

Options:
- `--skip-confirmation` - Skip confirmation prompts

#### Cross-Encoder Inference
Runs inference for a single cross-encoder:

```bash
./scripts/infer_cross.sh "castorini/monot5-base-msmarco"
./scripts/infer_cross.sh "castorini/monot5-3b-msmarco"
```

### Fusion Analysis

Runs fusion analysis combining multiple model outputs with different fusion methods:

```bash
./scripts/analyze.sh <year> <data_split> [experiment_type] [nsf_grid_search] [weight_resolution]
```

Examples:
```bash
./scripts/analyze.sh 2025 test
./scripts/analyze.sh 2025 test full true 0.1  # With NSF grid search
```

Arguments:
- `year` - Dataset year (`2024` or `2025`)
- `data_split` - Data split (`train`, `dev`, or `test`)
- `experiment_type` - Experiment type (default: `full`)
- `nsf_grid_search` - Enable NSF grid search (`true`/`false`)
- `weight_resolution` - Weight resolution for grid search

#### Fusion Methods
- **NSF** - Normalized Score Fusion with normalization options (min-max, z-score, percentile-rank)
- **BCF** - Borda Count Fusion
- **RRF** - Reciprocal Rank Fusion

## Configuration

All model configurations are managed via YAML files in the `configs/` directory.

### Bi-Encoder Configuration (`configs/bi_encoder.yaml`)

```yaml
google-bert/bert-base-multilingual-cased:
  ARCHITECTURE: "bi_encoder"
  PRETRAINED_MODEL: "google-bert/bert-base-multilingual-cased"
  TOKENIZER_PATH: "google-bert/bert-base-multilingual-cased"
  TRAINING_SAMPLES_FILE: "dataset/task2_train_negatives_2025.json"

  LOSS_FUNCTION: "contrastive"
  AUGMENTATION_TYPE: "standard"
  DYNAMIC_SAMPLING: true
  DYNAMIC_SAMPLING_STRATEGY: "hard"
  MAX_NEGATIVES_PER_POSITIVE: 5

  NUM_EPOCHS: 3
  TRAIN_BATCH_SIZE: 32
  EVAL_BATCH_SIZE: 32
  GRADIENT_ACCUMULATION_STEPS: 1
  OUTPUT_DIR: "checkpoints"
  IS_FP16: true
  YEAR: "2025"
```

### Cross-Encoder Configuration (`configs/cross_encoder.yaml`)

```yaml
castorini/monot5-base-msmarco:
  ARCHITECTURE: "cross_encoder"
  PRETRAINED_MODEL: "castorini/monot5-base-msmarco"
  TOKENIZER_PATH: "castorini/monot5-base-msmarco"
  TRAINING_SAMPLES_FILE: "dataset/task2_train_negatives_2025.json"

  LOSS_FUNCTION: "cross_entropy"
  MODEL_TYPE: "seq2seq"
  DYNAMIC_SAMPLING_STRATEGY: "hard"
  MAX_NEGATIVES_PER_POSITIVE: 5

  NUM_EPOCHS: 3
  TRAIN_BATCH_SIZE: 2
  EVAL_BATCH_SIZE: 2
  OUTPUT_DIR: "checkpoints"
  IS_FP16: true
  YEAR: "2025"
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `ARCHITECTURE` | Model architecture (`bi_encoder` or `cross_encoder`) |
| `LOSS_FUNCTION` | Training loss (`contrastive`, `multiple_negatives_ranking`, `cross_entropy`) |
| `DYNAMIC_SAMPLING` | Enable dynamic negative sampling |
| `DYNAMIC_SAMPLING_STRATEGY` | Sampling strategy (`hard`, `random`) |
| `MAX_NEGATIVES_PER_POSITIVE` | Number of negatives per positive sample |
| `MODEL_TYPE` | Cross-encoder model type (`seq2seq` for T5 models) |
| `IS_FP16` | Enable mixed precision training |

## Output

### Checkpoints
Trained models are saved to `checkpoints/<model-name>/`

### Results
- `fusion_results/` - Fusion analysis results
- `single_model_results/` - Individual model evaluation results

## License

This project is provided for research and educational purposes.
