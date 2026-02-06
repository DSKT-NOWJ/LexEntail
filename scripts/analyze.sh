#!/bin/bash
# Fusion analysis script for LCE framework.
#
# This script runs fusion analysis with different combinations of models,
# fusion methods, and normalization techniques adapted for the LCE codebase structure.

# Activate virtual environment
source env/bin/activate

# Validate arguments
YEAR=$1
if [ "$YEAR" != "2024" ] && [ "$YEAR" != "2025" ]; then
    echo "ERROR: First argument corresponds to the dataset year and must be either '2024' or '2025'."
    echo "Usage: $0 <year> <data_split> [experiment_type] [nsf_grid_search] [weight_resolution]"
    echo "Example: $0 2025 test"
    echo "Example: $0 2025 test full true 0.1  # Enable NSF grid search"
    exit 1
fi

DATA_SPLIT=$2
if [ "$DATA_SPLIT" != "test" ] && [ "$DATA_SPLIT" != "dev" ] && [ "$DATA_SPLIT" != "train" ]; then
    echo "ERROR: Second argument corresponds to the data split and must be 'train', 'dev', or 'test'."
    echo "Usage: $0 <year> <data_split> [experiment_type] [nsf_grid_search] [weight_resolution]"
    echo "Example: $0 2025 test"
    echo "Example: $0 2025 test full true 0.1  # Enable NSF grid search"
    exit 1
fi

EXPERIMENT_TYPE=${3:-"full"}
if [ "$EXPERIMENT_TYPE" != "full" ]; then
    echo "ERROR: Third argument (optional) corresponds to experiment type: 'full'."
    echo "  - full: Run all models and all fusion combinations"
    echo "Usage: $0 <year> <data_split> [experiment_type] [nsf_grid_search] [weight_resolution]"
    echo "Example: $0 2025 test full"
    echo "Example: $0 2025 test full true 0.1  # Enable NSF grid search"
    exit 1
fi

# Logging disabled for best performance


MODEL_COMBINATIONS=(
    # Single models
    # "--run_bert"
    # "--run_xlm_roberta"
    # "--run_e5"
    # "--run_bge"
    # "--run_monot5"
    # "--run_monot5_3b"

    #Chap 2
    "--run_bert --run_xlm_roberta"
    "--run_bert --run_e5"
    "--run_bert --run_bge"
    "--run_bert --run_monot5"
    "--run_bert --run_monot5_3b"
    "--run_xlm_roberta --run_e5"
    "--run_xlm_roberta --run_bge"
    "--run_xlm_roberta --run_monot5"
    "--run_xlm_roberta --run_monot5_3b"
    "--run_e5 --run_bge"
    "--run_e5 --run_monot5"
    "--run_e5 --run_monot5_3b"
    "--run_bge --run_monot5"
    "--run_bge --run_monot5_3b"
    "--run_monot5 --run_monot5_3b"


    #Chap 3
    "--run_bert --run_xlm_roberta --run_e5"
    "--run_bert --run_xlm_roberta --run_bge"
    "--run_bert --run_xlm_roberta --run_monot5"
    "--run_bert --run_xlm_roberta --run_monot5_3b"
    "--run_bert --run_e5 --run_bge"
    "--run_bert --run_e5 --run_monot5"
    "--run_bert --run_e5 --run_monot5_3b"
    "--run_bert --run_bge --run_monot5"
    "--run_bert --run_bge --run_monot5_3b"
    "--run_bert --run_monot5 --run_monot5_3b"
    "--run_xlm_roberta --run_e5 --run_bge"
    "--run_xlm_roberta --run_e5 --run_monot5"
    "--run_xlm_roberta --run_e5 --run_monot5_3b"
    "--run_xlm_roberta --run_bge --run_monot5"
    "--run_xlm_roberta --run_bge --run_monot5_3b"
    "--run_xlm_roberta --run_monot5 --run_monot5_3b"
    "--run_e5 --run_bge --run_monot5"
    "--run_e5 --run_bge --run_monot5_3b"
    "--run_e5 --run_monot5 --run_monot5_3b"
    "--run_bge --run_monot5 --run_monot5_3b"

    #Chap 4
    "--run_bert --run_xlm_roberta --run_e5 --run_bge"
    "--run_bert --run_xlm_roberta --run_e5 --run_monot5"
    "--run_bert --run_xlm_roberta --run_e5 --run_monot5_3b"
    "--run_bert --run_xlm_roberta --run_bge --run_monot5"
    "--run_bert --run_xlm_roberta --run_bge --run_monot5_3b"
    "--run_bert --run_xlm_roberta --run_monot5 --run_monot5_3b"
    "--run_bert --run_e5 --run_bge --run_monot5"
    "--run_bert --run_e5 --run_bge --run_monot5_3b"
    "--run_bert --run_e5 --run_monot5 --run_monot5_3b"
    "--run_bert --run_bge --run_monot5 --run_monot5_3b"
    "--run_xlm_roberta --run_e5 --run_bge --run_monot5"
    "--run_xlm_roberta --run_e5 --run_bge --run_monot5_3b"
    "--run_xlm_roberta --run_e5 --run_monot5 --run_monot5_3b"
    "--run_xlm_roberta --run_bge --run_monot5 --run_monot5_3b"
    "--run_e5 --run_bge --run_monot5 --run_monot5_3b"

    #Chap 5
    "--run_bert --run_xlm_roberta --run_e5 --run_bge --run_monot5"
    "--run_bert --run_xlm_roberta --run_e5 --run_bge --run_monot5_3b"
    "--run_bert --run_xlm_roberta --run_e5 --run_monot5 --run_monot5_3b"
    "--run_bert --run_xlm_roberta --run_bge --run_monot5 --run_monot5_3b"
    "--run_bert --run_e5 --run_bge --run_monot5 --run_monot5_3b"
    "--run_xlm_roberta --run_e5 --run_bge --run_monot5 --run_monot5_3b"

    # All models
    "--run_bert --run_xlm_roberta --run_e5 --run_bge --run_monot5 --run_monot5_3b"
)

# Define fusion methods and normalizers
FUSIONERS=("nsf" "bcf" "rrf")
NORMALIZERS=("min-max" "z-score" "percentile-rank")

# NSF Grid Search Parameters
NSF_GRID_SEARCH=${4:-"false"}
WEIGHT_RESOLUTION=${5:-"0.1"}

# Function to get config value from YAML file
get_config_value() {
    local config_file=$1
    local model_name=$2
    local key=$3

    # Use Python to parse YAML and extract the value
    python3 -c "
import yaml
import sys

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)

    if '$model_name' in config and '$key' in config['$model_name']:
        print(config['$model_name']['$key'])
    else:
        print('3' if '$key' == 'MAX_NEGATIVES_PER_POSITIVE' else 'hard')
except Exception:
    print('3' if '$key' == 'MAX_NEGATIVES_PER_POSITIVE' else 'hard')
"
}

# Function to get model-specific config values
get_model_config() {
    local models=$1
    local key=$2
    local default_value=$3

    # Extract the first model from the combination to get config values
    local first_model=$(echo $models | awk '{print $1}')

    # Map model flags to actual model names and config files
    local model_name=""
    local config_file=""

    case "$first_model" in
        "--run_bert")
            model_name="google-bert/bert-base-multilingual-cased"
            config_file="configs/bi_encoder.yaml"
            ;;
        "--run_xlm_roberta")
            model_name="FacebookAI/xlm-roberta-large"
            config_file="configs/bi_encoder.yaml"
            ;;
        "--run_e5")
            model_name="intfloat/multilingual-e5-large"
            config_file="configs/bi_encoder.yaml"
            ;;
        "--run_bge")
            model_name="BAAI/bge-m3"
            config_file="configs/bi_encoder.yaml"
            ;;
        "--run_monot5")
            model_name="castorini/monot5-base-msmarco"
            config_file="configs/cross_encoder.yaml"
            ;;
        "--run_monot5_3b")
            model_name="castorini/monot5-3b-msmarco"
            config_file="configs/cross_encoder.yaml"
            ;;
        *)
            echo "$default_value"
            return
            ;;
    esac

    local value=$(get_config_value "$config_file" "$model_name" "$key")
    echo "$value"
}

# Function to check if models are available (silent mode for performance)
check_model_availability() {
    local models_missing=0
    
    if [ "$EXPERIMENT_TYPE" != "" ]; then
        # Check for BM25 index
        if [ ! -d "dataset/bm25_indexes/coliee_task2/$YEAR/$DATA_SPLIT" ]; then
            models_missing=1
        fi
        
        # Check for trained model checkpoints
        models=("bert-base-multilingual-cased" "xlm-roberta-large" "multilingual-e5-large" "bge-m3" "monot5-base-msmarco" "monot5-3b-msmarco")
        for model in "${models[@]}"; do
            if [ ! -d "checkpoints/${model}" ]; then
                models_missing=1
            fi
        done
    fi
}

# Function to run single model evaluation
run_single_model_evaluation() {
    local models=$1

    # Get config values for this model
    local sampling_strategy=$(get_model_config "$models" "DYNAMIC_SAMPLING_STRATEGY" "hard")
    local max_negatives=$(get_model_config "$models" "MAX_NEGATIVES_PER_POSITIVE" "3")

    # Create output directory
    output_dir="single_model_results/${YEAR}_${DATA_SPLIT}"

    # Build command
    cmd="python src/analyze.py --year $YEAR --data_split $DATA_SPLIT $models --output_dir $output_dir --dynamic_sampling_strategy $sampling_strategy --max_negatives_per_positive $max_negatives"

    # Execute command silently
    eval $cmd 2>/dev/null
}

# Function to run NSF grid search
run_nsf_grid_search() {
    local models=$1

    # Get config values for the first model in the combination
    local sampling_strategy=$(get_model_config "$models" "DYNAMIC_SAMPLING_STRATEGY" "hard")
    local max_negatives=$(get_model_config "$models" "MAX_NEGATIVES_PER_POSITIVE" "3")

    # Create output directory
    output_dir="fusion_results/${YEAR}_${DATA_SPLIT}"

    # Build command
    cmd="python src/analyze.py --year $YEAR --data_split $DATA_SPLIT $models --fusion nsf --nsf_grid_search --weight_resolution $WEIGHT_RESOLUTION --output_dir $output_dir --dynamic_sampling_strategy $sampling_strategy --max_negatives_per_positive $max_negatives"

    # Execute command silently
    eval $cmd 2>/dev/null
}

# Function to run fusion analysis
run_fusion_analysis() {
    local models=$1
    local fusion=$2
    local normalization=$3

    # Get config values for the first model in the combination
    local sampling_strategy=$(get_model_config "$models" "DYNAMIC_SAMPLING_STRATEGY" "hard")
    local max_negatives=$(get_model_config "$models" "MAX_NEGATIVES_PER_POSITIVE" "3")

    # Create output directory
    output_dir="fusion_results/${YEAR}_${DATA_SPLIT}"

    # Build command
    cmd="python src/analyze.py --year $YEAR --data_split $DATA_SPLIT $models --fusion $fusion --output_dir $output_dir --dynamic_sampling_strategy $sampling_strategy --max_negatives_per_positive $max_negatives"

    # Add normalization for NSF (only if specific normalization is requested)
    if [ "$fusion" = "nsf" ] && [ "$normalization" != "all" ]; then
        cmd="$cmd --normalization $normalization"
    fi

    # Execute command silently
    eval $cmd 2>/dev/null
}

# Main execution
main() {
    # Check model availability
    check_model_availability
    
    # Run all combinations
    for models in "${MODEL_COMBINATIONS[@]}"; do
        # Count number of models in this combination
        model_count=$(echo $models | wc -w)

        if [ $model_count -eq 1 ]; then
            # Single model - no fusion needed
            run_single_model_evaluation "$models"
        else
            # Multiple models - run fusion
            for fusion in "${FUSIONERS[@]}"; do
                if [ "$fusion" = "nsf" ] && [ "$NSF_GRID_SEARCH" = "true" ]; then
                    # NSF Grid Search Mode
                    run_nsf_grid_search "$models"
                elif [ "$fusion" = "nsf" ]; then
                    # Regular NSF - let analyze.py handle all normalizations internally
                    run_fusion_analysis "$models" "$fusion" "all"
                else
                    # BCF and RRF don't use normalization
                    run_fusion_analysis "$models" "$fusion" "none"
                fi
            done
        fi
    done
}

# Create output directories
mkdir -p fusion_results
# mkdir -p single_model_results
# mkdir -p nsf_grid_search_results

# Run main function
main