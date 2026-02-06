#!/bin/bash

# Comprehensive inference script for all bi-encoders
# Usage: ./infer_bi.sh [--skip-confirmation]

source env/bin/activate

set -e

# Parse command line arguments
SKIP_CONFIRMATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-confirmation)
            SKIP_CONFIRMATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-confirmation]"
            echo ""
            echo "Options:"
            echo "  --skip-confirmation Skip confirmation prompts"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Runs inference for all bi-encoder models from configs/bi_encoder.yaml"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to get models from YAML config
get_models_from_config() {
    local config_file="$1"
    python3 -c "import yaml; config = yaml.safe_load(open('$config_file')); print(' '.join(config.keys()))"
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
ARCHITECTURE="bi_encoder"
LOSS_FUNCTION="contrastive"
AUGMENTATION_TYPE="standard"
DYNAMIC_SAMPLING="True"
DYNAMIC_SAMPLING_STRATEGY="hard"
MAX_NEGATIVES_PER_POSITIVE=3

PRETRAINED_MODEL="google-bert/bert-base-multilingual-cased"
TOKENIZER_PATH="google-bert/bert-base-multilingual-cased"
NUM_EPOCHS=5
TRAIN_BATCH_SIZE=48
EVAL_BATCH_SIZE=48
GRADIENT_ACCUMULATION_STEPS=1
LOGGING_STRATEGY="steps"
LOGGING_STEPS=50
OUTPUT_DIR="checkpoints"

# Inference specific defaults
YEAR="2025"
CHECKPOINT_STRATEGY="best"
MANUAL_CHECKPOINT_PATH=""

# Function to display inference summary
display_inference_summary() {
    local model_name="$1"
    
    echo ""
    echo "=== Inference Configuration for $model_name ==="
    echo "Architecture              : $ARCHITECTURE"
    echo "Pretrained Model         : $PRETRAINED_MODEL"
    echo "Output Directory         : $OUTPUT_DIR"
    echo "Year                     : $YEAR"
    echo "Max Negatives/Positive   : $MAX_NEGATIVES_PER_POSITIVE"
    echo "Dynamic Sampling Strategy: $DYNAMIC_SAMPLING_STRATEGY"
    echo ""
}

# Function to run inference for a single model
infer_model() {
    local model_name="$1"
    local config_file="$2"
    
    echo ""
    echo "=========================================="
    echo "Running Inference: $model_name"
    echo "=========================================="
    
    # Load configuration
    if ! load_yaml_config "$model_name" "$config_file"; then
        echo "Failed to load configuration for $model_name"
        return 1
    fi
    
    # Map training variables to inference variables if needed
    if [ -n "$PRETRAINED_MODEL" ] && [ -z "$PRETRAINED_MODEL_PATH" ]; then
        PRETRAINED_MODEL_PATH="$PRETRAINED_MODEL"
    fi
    
    # Display configuration
    display_inference_summary "$model_name"
    
    # Check if required values are set
    if [ -z "$OUTPUT_DIR" ]; then
        echo "Error: OUTPUT_DIR not set"
        return 1
    fi
    
    if [ -z "$PRETRAINED_MODEL_PATH" ]; then
        echo "Error: PRETRAINED_MODEL_PATH not set"
        return 1
    fi
    
    if [ -z "$YEAR" ]; then
        echo "Error: YEAR not set"
        return 1
    fi
    
    # Set checkpoint path
    echo "=== Setting Checkpoint Path ==="
    # Extract model name for checkpoint directory
    MODEL_DIR_NAME=$(echo "$PRETRAINED_MODEL" | sed 's/.*\///')
    CHECKPOINT_PATH="$OUTPUT_DIR/$MODEL_DIR_NAME"
    
    echo "Using checkpoint directory: $CHECKPOINT_PATH"
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "Error: Checkpoint directory $CHECKPOINT_PATH does not exist"
        return 1
    fi
    
    echo "Final checkpoint path: $CHECKPOINT_PATH"
    echo ""
    
    echo "Inference started at: $(date)"
    
    # Run inference
    echo "=== Starting Inference ==="
    echo "Using pretrained model: $PRETRAINED_MODEL for output naming"
    python src/infer_bi.py \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --pretrained_model "$PRETRAINED_MODEL" \
        --tokenizer_path "$CHECKPOINT_PATH" \
        --year "$YEAR" \
        --dynamic_sampling_strategy "$DYNAMIC_SAMPLING_STRATEGY" \
        --max_negatives_per_positive "$MAX_NEGATIVES_PER_POSITIVE" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --dataset_path "dataset" \
        --segment "test" \
        --approach "per_case" \
        --device "cuda"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Inference completed successfully for $model_name"
        echo "Inference finished at: $(date)"
    else
        echo ""
        echo "✗ Inference failed for $model_name"
        echo "Exit code: $exit_code"
        echo "Inference failed at: $(date)"
        return $exit_code
    fi
}

# Main inference function
main() {
    echo "=== Comprehensive Bi-Encoder Inference Script Started ==="
    echo "Started at: $(date)"
    
    # Check if inference script exists
    if [ ! -f "src/infer_bi.py" ]; then
        echo "Error: src/infer_bi.py not found"
        echo "Make sure you're running this script from the project root directory"
        exit 1
    fi
    
    local total_models=0
    local successful_models=0
    local failed_models=()
    
    # Get models from config file
    if [ -f "configs/bi_encoder.yaml" ]; then
        BI_MODELS=($(get_models_from_config "configs/bi_encoder.yaml"))
        echo "Found ${#BI_MODELS[@]} bi-encoder models: ${BI_MODELS[*]}"
        total_models=${#BI_MODELS[@]}
    else
        echo "Error: configs/bi_encoder.yaml not found"
        exit 1
    fi
    
    if [ $total_models -eq 0 ]; then
        echo "Error: No models found to infer"
        exit 1
    fi
    
    echo ""
    echo "=== Inference Plan ==="
    echo "Total models to infer: $total_models"
    
    if [ "$SKIP_CONFIRMATION" != true ]; then
        echo ""
        echo "This will run inference for all models sequentially."
        echo "Press Enter to continue or Ctrl+C to cancel..."
        read -r
    fi
    
    # Run inference for all models
    echo ""
    echo "=========================================="
    echo "Starting Bi-Encoder Inference Phase"
    echo "=========================================="
    
    for model in "${BI_MODELS[@]}"; do
        if infer_model "$model" "configs/bi_encoder.yaml"; then
            successful_models=$((successful_models + 1))
        else
            failed_models+=("$model")
        fi
    done
    
    # Final summary
    echo ""
    echo "=========================================="
    echo "Inference Summary"
    echo "=========================================="
    echo "Completed at: $(date)"
    echo "Total models: $total_models"
    echo "Successful: $successful_models"
    echo "Failed: ${#failed_models[@]}"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        echo ""
        echo "Failed models:"
        for model in "${failed_models[@]}"; do
            echo "  ✗ $model"
        done
        echo ""
        echo "Inference completed with failures"
        exit 1
    else
        echo ""
        echo "✓ All models inferred successfully!"
        exit 0
    fi
}

# Run main function
main "$@"