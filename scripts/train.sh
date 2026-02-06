#!/bin/bash

# Comprehensive training script for all bi-encoders and cross-encoders
# Usage: ./train.sh [--bi-only] [--cross-only] [--skip-confirmation]

source env/bin/activate

set -e  # Exit on any error

# Parse command line arguments
BI_ONLY=false
CROSS_ONLY=false
SKIP_CONFIRMATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --bi-only)
            BI_ONLY=true
            shift
            ;;
        --cross-only)
            CROSS_ONLY=true
            shift
            ;;
        --skip-confirmation)
            SKIP_CONFIRMATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--bi-only] [--cross-only] [--skip-confirmation]"
            echo ""
            echo "Options:"
            echo "  --bi-only          Train only bi-encoders"
            echo "  --cross-only       Train only cross-encoders"
            echo "  --skip-confirmation Skip confirmation prompts"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Without options, trains all models from both config files"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate that both flags aren't set
if [ "$BI_ONLY" = true ] && [ "$CROSS_ONLY" = true ]; then
    echo "Error: Cannot use both --bi-only and --cross-only flags"
    exit 1
fi

# Function to get models from YAML config
get_models_from_config() {
    local config_file="$1"
    python3 -c "import yaml; config = yaml.safe_load(open('$config_file')); print(' '.join(config.keys()))"
}

# Function to load YAML configuration (similar to individual scripts)
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

# Function to display training summary
display_training_summary() {
    local architecture="$1"
    local model_name="$2"
    
    echo ""
    echo "=== Training Configuration for $model_name ===" 
    echo "Architecture              : $ARCHITECTURE"
    echo "Loss Function            : $LOSS_FUNCTION"
    echo "Pretrained Model         : $PRETRAINED_MODEL"
    echo "Number of Epochs         : $NUM_EPOCHS"
    echo "Train Batch Size         : $TRAIN_BATCH_SIZE"
    echo "Output Directory         : $OUTPUT_DIR"
    
    if [ "$architecture" = "bi_encoder" ]; then
        echo "Augmentation Type        : $AUGMENTATION_TYPE"
        echo "Dynamic Sampling         : $DYNAMIC_SAMPLING"
        echo "Dynamic Sampling Strategy: $DYNAMIC_SAMPLING_STRATEGY"
    elif [ "$architecture" = "cross_encoder" ]; then
        echo "Model Type               : $MODEL_TYPE"
    fi
    
    echo "Max Negatives/Positive   : $MAX_NEGATIVES_PER_POSITIVE"
    echo ""
}

# Function to train a single model
train_model() {
    local model_name="$1"
    local config_file="$2"
    local architecture="$3"
    
    echo ""
    echo "=========================================="
    echo "Training $architecture: $model_name"
    echo "=========================================="
    
    # Load configuration
    if ! load_yaml_config "$model_name" "$config_file"; then
        echo "Failed to load configuration for $model_name"
        return 1
    fi
    
    # Display configuration
    display_training_summary "$architecture" "$model_name"
    
    # Create output directory if it doesn't exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi
    
    echo "Training started at: $(date)"
    
    # Build command based on architecture
    if [ "$architecture" = "bi_encoder" ]; then
        CMD="python src/train.py \
            --architecture \"$ARCHITECTURE\" \
            --loss_function \"$LOSS_FUNCTION\" \
            --augmentation_type \"$AUGMENTATION_TYPE\" \
            --dynamic_sampling_strategy \"$DYNAMIC_SAMPLING_STRATEGY\" \
            --max_negatives_per_positive \"$MAX_NEGATIVES_PER_POSITIVE\" \
            --training_samples_file \"$TRAINING_SAMPLES_FILE\" \
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
    else
        # cross_encoder
        CMD="python src/train.py \
            --architecture \"$ARCHITECTURE\" \
            --loss_function \"$LOSS_FUNCTION\" \
            --pretrained_model \"$PRETRAINED_MODEL\" \
            --tokenizer_path \"$TOKENIZER_PATH\" \
            --model_type \"$MODEL_TYPE\" \
            --dynamic_sampling_strategy \"$DYNAMIC_SAMPLING_STRATEGY\" \
            --max_negatives_per_positive \"$MAX_NEGATIVES_PER_POSITIVE\" \
            --training_samples_file \"$TRAINING_SAMPLES_FILE\" \
            --num_epochs \"$NUM_EPOCHS\" \
            --train_batch_size \"$TRAIN_BATCH_SIZE\" \
            --eval_batch_size \"$EVAL_BATCH_SIZE\" \
            --gradient_accumulation_steps \"$GRADIENT_ACCUMULATION_STEPS\" \
            --logging_strategy \"$LOGGING_STRATEGY\" \
            --logging_steps \"$LOGGING_STEPS\" \
            --output_dir \"$OUTPUT_DIR\" \
            --year \"$YEAR\""
    fi
    
    # Add FP16 flag if enabled
    if [ "$IS_FP16" = "true" ]; then
        CMD="$CMD --fp16"
    fi
    
    # Execute the command
    eval $CMD
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo " Training completed successfully for $model_name"
        echo "Training finished at: $(date)"
        echo "Model checkpoints saved in: $OUTPUT_DIR"
    else
        echo ""
        echo " Training failed for $model_name"
        echo "Exit code: $exit_code"
        echo "Training failed at: $(date)"
        return $exit_code
    fi
}

# Main training function
main() {
    echo "=== Comprehensive Model Training Script Started ==="
    echo "Started at: $(date)"
    
    # Check if training script exists
    if [ ! -f "src/train.py" ]; then
        echo "Error: src/train.py not found"
        echo "Make sure you're running this script from the project root directory"
        exit 1
    fi
    
    local total_models=0
    local successful_models=0
    local failed_models=()
    
    # Get models from config files
    if [ "$CROSS_ONLY" != true ]; then
        if [ -f "configs/bi_encoder.yaml" ]; then
            BI_MODELS=($(get_models_from_config "configs/bi_encoder.yaml"))
            echo "Found ${#BI_MODELS[@]} bi-encoder models: ${BI_MODELS[*]}"
            total_models=$((total_models + ${#BI_MODELS[@]}))
        else
            echo "Warning: configs/bi_encoder.yaml not found"
        fi
    fi
    
    if [ "$BI_ONLY" != true ]; then
        if [ -f "configs/cross_encoder.yaml" ]; then
            CROSS_MODELS=($(get_models_from_config "configs/cross_encoder.yaml"))
            echo "Found ${#CROSS_MODELS[@]} cross-encoder models: ${CROSS_MODELS[*]}"
            total_models=$((total_models + ${#CROSS_MODELS[@]}))
        else
            echo "Warning: configs/cross_encoder.yaml not found"
        fi
    fi
    
    if [ $total_models -eq 0 ]; then
        echo "Error: No models found to train"
        exit 1
    fi
    
    echo ""
    echo "=== Training Plan ==="
    echo "Total models to train: $total_models"
    
    if [ "$SKIP_CONFIRMATION" != true ]; then
        echo ""
        echo "This will train all models sequentially."
        echo "Press Enter to continue or Ctrl+C to cancel..."
        read -r
    fi
    
    # Train bi-encoders
    if [ "$CROSS_ONLY" != true ] && [ ${#BI_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "Starting Bi-Encoder Training Phase"
        echo "=========================================="
        
        for model in "${BI_MODELS[@]}"; do
            if train_model "$model" "configs/bi_encoder.yaml" "bi_encoder"; then
                successful_models=$((successful_models + 1))
            else
                failed_models+=("$model (bi-encoder)")
            fi
        done
    fi
    
    # Train cross-encoders
    if [ "$BI_ONLY" != true ] && [ ${#CROSS_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "=========================================="
        echo "Starting Cross-Encoder Training Phase"
        echo "=========================================="
        
        for model in "${CROSS_MODELS[@]}"; do
            if train_model "$model" "configs/cross_encoder.yaml" "cross_encoder"; then
                successful_models=$((successful_models + 1))
            else
                failed_models+=("$model (cross-encoder)")
            fi
        done
    fi
    
    # Final summary
    echo ""
    echo "=========================================="
    echo "Training Summary"
    echo "=========================================="
    echo "Completed at: $(date)"
    echo "Total models: $total_models"
    echo "Successful: $successful_models"
    echo "Failed: ${#failed_models[@]}"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        echo ""
        echo "Failed models:"
        for model in "${failed_models[@]}"; do
            echo "   $model"
        done
        echo ""
        echo "Training completed with failures"
        exit 1
    else
        echo ""
        echo " All models trained successfully!"
        exit 0
    fi
}

# Run main function
main "$@"