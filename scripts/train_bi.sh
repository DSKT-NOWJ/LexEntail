#!/bin/bash

# Enhanced bi-encoder training script with YAML configuration management
# Usage: ./train_bi.sh [model_name]
# Example: ./train_bi.sh "google-bert/bert-base-multilingual-cased"
source env/bin/activate

set -e  # Exit on any error

# Function to print usage
print_usage() {
    echo "Usage: $0 [model_name]"
    echo "Example: $0 \"google-bert/bert-base-multilingual-cased\""
    echo "Example: $0 \"FacebookAI/xlm-roberta-large\""
    echo "Example: $0 \"intfloat/multilingual-e5-large\""
    echo "Example: $0 \"BAAI/bge-m3\""
    echo ""
    echo "Available models in configs/bi_encoder.yaml:"
    python3 -c "import yaml; config = yaml.safe_load(open('configs/bi_encoder.yaml')); [print(f'  - {model}') for model in config.keys()]"
}

# Function to load YAML configuration
load_yaml_config() {
    local model_name="$1"
    local config_file="configs/bi_encoder.yaml"
    
    echo "=== Loading Configuration ==="
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        echo "Error: Configuration file '$config_file' not found"
        exit 1
    fi
    
    # Check if training script exists
    if [ ! -f "src/train.py" ]; then
        echo "Error: src/train.py not found"
        echo "Make sure you're running this script from the project root directory"
        exit 1
    fi
    
    # Load configuration using Python
    echo "Loading configuration for model: $model_name"
    
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
    
    echo "✓ Configuration loaded successfully"
}

# Function to display configuration
display_config() {
    echo ""
    echo "=== Training Configuration ==="
    echo "Architecture              : $ARCHITECTURE"
    echo "Loss Function            : $LOSS_FUNCTION"
    echo "Augmentation Type        : $AUGMENTATION_TYPE"
    echo "Dynamic Sampling         : $DYNAMIC_SAMPLING"
    echo "Dynamic Sampling Strategy: $DYNAMIC_SAMPLING_STRATEGY"
    echo "Max Negatives/Positive   : $MAX_NEGATIVES_PER_POSITIVE"
    echo ""
    echo "=== Model Configuration ==="
    echo "Pretrained Model         : $PRETRAINED_MODEL"
    echo "Tokenizer Path           : $TOKENIZER_PATH"
    echo ""
    echo "=== Training Parameters ==="
    echo "Number of Epochs         : $NUM_EPOCHS"
    echo "Train Batch Size         : $TRAIN_BATCH_SIZE"
    echo "Eval Batch Size          : $EVAL_BATCH_SIZE"
    echo "Gradient Accumulation    : $GRADIENT_ACCUMULATION_STEPS"
    echo "Logging Strategy         : $LOGGING_STRATEGY"
    echo "Logging Steps            : $LOGGING_STEPS"
    echo "Output Directory         : $OUTPUT_DIR"
    echo ""
}

# Load configuration
echo "=== Bi-Encoder Training Script Started ==="

# Check arguments
if [ $# -gt 1 ]; then
    echo "Error: Too many arguments"
    print_usage
    exit 1
fi

if [ $# -eq 1 ]; then
    MODEL_NAME="$1"
else
    echo "Error: Model name is required"
    print_usage
    exit 1
fi

# Load configuration from YAML
load_yaml_config "$MODEL_NAME"

# Display configuration
display_config

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
echo "=== Computed Values ==="
echo "Effective Batch Size     : $EFFECTIVE_BATCH_SIZE"
echo ""

# Confirm before starting training
echo "=== Ready to Start Training ==="
echo "Press Enter to continue or Ctrl+C to cancel..."
read -r

# Run training
echo "=== Starting Training ==="
echo "Training started at: $(date)"

# Build the command with conditional flags
CMD="python src/train.py \
    --architecture \"$ARCHITECTURE\" \
    --loss_function \"$LOSS_FUNCTION\" \
    --augmentation_type \"$AUGMENTATION_TYPE\" \
    --dynamic_sampling_strategy \"$DYNAMIC_SAMPLING_STRATEGY\" \
    --max_negatives_per_positive \"$MAX_NEGATIVES_PER_POSITIVE\" \
    --pretrained_model \"$PRETRAINED_MODEL\" \
    --tokenizer_path \"$TOKENIZER_PATH\" \
    --num_epochs \"$NUM_EPOCHS\" \
    --train_batch_size \"$TRAIN_BATCH_SIZE\" \
    --eval_batch_size \"$EVAL_BATCH_SIZE\" \
    --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" \
    --logging_strategy \"$LOGGING_STRATEGY\" \
    --logging_steps \"$LOGGING_STEPS\" \
    --output_dir \"$OUTPUT_DIR\" \
    --year \"$YEAR\""

# Add boolean flags only if they are true
if [ "$DYNAMIC_SAMPLING" = "true" ]; then
    CMD="$CMD --dynamic_sampling"
fi

if [ "$IS_FP16" = "true" ]; then
    CMD="$CMD --fp16"
fi

# Execute the command
eval $CMD

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Training Completed Successfully ==="
    echo "Training finished at: $(date)"
    echo "Model checkpoints saved in: $OUTPUT_DIR"
else
    echo ""
    echo "=== Training Failed ==="
    echo "Exit code: $TRAINING_EXIT_CODE"
    echo "Training failed at: $(date)"
    exit $TRAINING_EXIT_CODE
fi