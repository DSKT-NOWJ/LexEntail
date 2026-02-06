import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mode", help="Running mode", default="None")
parser.add_argument(
    "--pretrained_model", help="Pretrain model", default="bert-base-multilingual-cased"
)
parser.add_argument(
    "--tokenizer_path", help="Path to tokenizer", default="bert-base-multilingual-cased"
)
parser.add_argument(
    "--augmentation_type", help="Augmentation type", default="standard", type=str
)
parser.add_argument("--dynamic_sampling", help="Dynamic sampling", action="store_true")
parser.add_argument("--dynamic_sampling_strategy", help="Dynamic sampling strategy", default="random", type=str)
parser.add_argument("--max_negatives_per_positive", help="Maximum negatives per positive", default=10, type=int)

parser.add_argument("--freeze_mode", help="Freeze mode", default="None")
parser.add_argument("--checkpoint_path", help="Path checkpoint", default="default")
parser.add_argument("--training_samples_file", help="Path to training samples file", default="dataset/task2_train_negatives_2025.json")
# Model architecture arguments
parser.add_argument(
    "--architecture",
    help="Model architecture",
    choices=["bi_encoder", "cross_encoder"],
    default="cross_encoder",
)
parser.add_argument(
    "--model_type",
    help="Model type",
    choices=["seq2seq", "causal_lm", "bert", "e5", "bge"],
    default="causal_lm",
)
parser.add_argument(
    "--loss_function",
    help="Loss function",
    # choices=["cross_entropy", "contrastive", "weighted_cross_entropy"],
    default="cross_entropy",
)
# parser.add_argument("--temperature", help="Temperature for contrastive loss", default=0.07, type=float)
# parser.add_argument("--margin", help="Margin for contrastive loss", default=0.2, type=float)

parser.add_argument(
    "--train_batch_size", help="Batch size for training", default=8, type=int
)
parser.add_argument(
    "--eval_batch_size", help="Batch size for evaluation", default=8, type=int
)
parser.add_argument(
    "--num_epochs", help="Number of epoch for training", default=10, type=int
)
parser.add_argument(
    "--used_gpu", help="Number of gpu for training", default=0, type=int
)
parser.add_argument("--w_loss", help="Weight loss", default=0, type=float)
parser.add_argument("--learning_rate", help="Learning rate", default=5e-5, type=float)

parser.add_argument("--dataset_path", help="Path to dataset", default="dataset")
parser.add_argument("--negative_mode", help="Negative mode", default="hard")
parser.add_argument(
    "--negative_num", help="Number of negative sample", default=3, type=int
)
parser.add_argument("--fast_dev_run", help="Fast dev run", default=0)

parser.add_argument("--output_dir", help="Output directory", default="checkpoints")

parser.add_argument("--is_fp16", help="Is fp16", default=0, type=int)

parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--warmup_ratio", default=0.1, type=float)
parser.add_argument("--optim", default="adafactor", type=str)
parser.add_argument("--label_smoothing_factor", default=0.1, type=float)
parser.add_argument("--fp16", action="store_true")

parser.add_argument("--seed", default=442, type=int)
parser.add_argument("--logging_strategy", default="epoch", type=str)
parser.add_argument("--logging_steps", default=1, type=int)

parser.add_argument("--year", default="2025", type=str)

args = parser.parse_args()


MODE = args.mode
PRETRAINED_MODEL = args.pretrained_model
AUGMENTATION_TYPE = args.augmentation_type
DYNAMIC_SAMPLING = args.dynamic_sampling
DYNAMIC_SAMPLING_STRATEGY = args.dynamic_sampling_strategy
MAX_NEGATIVES_PER_POSITIVE = args.max_negatives_per_positive

FREEZE_MODE = args.freeze_mode
CHECKPOINT_PATH = args.checkpoint_path
TOKENIZER_PATH = args.tokenizer_path
TRAINING_SAMPLES_FILE = args.training_samples_file

# Architecture settings
ARCHITECTURE = args.architecture
MODEL_TYPE = args.model_type
LOSS_FUNCTION = args.loss_function

TRAIN_BATCH_SIZE = args.train_batch_size
EVAL_BATCH_SIZE = args.eval_batch_size
N_EPOCH = args.num_epochs
USED_GPU = args.used_gpu
W_LOSS = args.w_loss
LEARNING_RATE = args.learning_rate

DATASET_PATH = args.dataset_path

FAST_DEV_RUN = args.fast_dev_run
YEAR = args.year

IS_FP16 = args.is_fp16

OUTPUT_DIR = args.output_dir

GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
WEIGHT_DECAY = args.weight_decay
WARMUP_RATIO = args.warmup_ratio
OPTIM = args.optim
LABEL_SMOOTHING_FACTER = args.label_smoothing_factor
FP16 = args.fp16

SEED = args.seed

LOGGING_STRATEGY = args.logging_strategy
LOGGING_STEPS = args.logging_steps
