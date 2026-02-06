#!/bin/bash
source env/bin/activate

set -e

# Function to print usage
print_usage() {
    echo "Usage: $0 [model_name]"
    echo "Example: $0 castorini/monot5-base-msmarco"
    echo "This will load configuration from configs/cross_encoder.yaml"
}

# Function to load YAML configuration (similar to train.sh)
load_yaml_config() {
    local model_name="$1"
    local config_file="$2"
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo "Error: Configuration file '$config_file' not found"
        return 1
    fi
    
    # Export variables from YAML
    eval $(python3 -c "
import yaml
import sys

with open('$config_file', 'r') as f:
    config = yaml.safe_load(f)

if '$model_name' not in config:
    print('Error: Model \"$model_name\" not found in configuration file', file=sys.stderr)
    print('Available models:', list(config.keys()), file=sys.stderr)
    sys.exit(1)

model_config = config['$model_name']

for key, value in model_config.items():
    if isinstance(value, bool):
        print(f'export {key}={str(value).lower()}')
    elif isinstance(value, (int, float)):
        print(f'export {key}={value}')
    else:
        print(f'export {key}=\"{value}\"')
")
}

# Set defaults first
ARCHITECTURE="cross_encoder"
LOSS_FUNCTION="cross_entropy"
MODEL_TYPE="seq2seq"
MAX_NEGATIVES_PER_POSITIVE=10

PRETRAINED_MODEL="castorini/monot5-base-msmarco"
TOKENIZER_PATH="castorini/monot5-base-msmarco"
NUM_EPOCHS=3
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
LOGGING_STRATEGY="steps"
LOGGING_STEPS=100
OUTPUT_DIR="checkpoints"

# Inference specific defaults
YEAR="2025"
CHECKPOINT_STRATEGY="best"
MANUAL_CHECKPOINT_PATH=""

# Load configuration
CONFIG_FILE="configs/cross_encoder.yaml"
if [ $# -eq 1 ]; then
    MODEL_NAME="$1"
    echo "Loading configuration for model: $MODEL_NAME"
    echo "From config file: $CONFIG_FILE"
    
    if ! load_yaml_config "$MODEL_NAME" "$CONFIG_FILE"; then
        echo "Failed to load configuration for $MODEL_NAME"
        exit 1
    fi
else
    echo "Error: Model name required"
    print_usage
    exit 1
fi

# Map training variables to inference variables if needed
if [ -n "$PRETRAINED_MODEL" ] && [ -z "$PRETRAINED_MODEL_PATH" ]; then
    PRETRAINED_MODEL_PATH="$PRETRAINED_MODEL"
fi

# Debug: Show what we loaded
echo "=== Configuration ==="
echo "Architecture: $ARCHITECTURE"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $PRETRAINED_MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Model Type: $MODEL_TYPE"
echo "Max Negatives Per Positive: $MAX_NEGATIVES_PER_POSITIVE"
echo "Year: $YEAR"
echo "Strategy: $CHECKPOINT_STRATEGY"
echo ""

# Check if required values are set
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: OUTPUT_DIR not set"
    exit 1
fi

if [ -z "$PRETRAINED_MODEL_PATH" ]; then
    echo "Error: PRETRAINED_MODEL_PATH not set"
    exit 1
fi

if [ -z "$YEAR" ]; then
    echo "Error: YEAR not set"
    exit 1
fi

# Set checkpoint path
echo "=== Setting Checkpoint Path ==="
# Extract model name for checkpoint directory
MODEL_DIR_NAME=$(echo "$PRETRAINED_MODEL" | sed 's/.*\///')
CHECKPOINT_PATH="$OUTPUT_DIR/$MODEL_DIR_NAME"

echo "Using checkpoint directory: $CHECKPOINT_PATH"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory $CHECKPOINT_PATH does not exist"
    exit 1
fi

echo "Final checkpoint path: $CHECKPOINT_PATH"
echo ""

# Run inference
echo "=== Starting Cross-Encoder Inference ==="
python src/infer_cross.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --pretrained_model "$PRETRAINED_MODEL_PATH" \
    --tokenizer_path "$CHECKPOINT_PATH" \
    --year "$YEAR" \
    --model_type "$MODEL_TYPE" \
    --dynamic_sampling_strategy "$DYNAMIC_SAMPLING_STRATEGY" \
    --max_negatives_per_positive "$MAX_NEGATIVES_PER_POSITIVE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --dataset_path "dataset" \
    --segment "test" \
    --approach "per_case" \
    --device "cuda"

echo "Done!"