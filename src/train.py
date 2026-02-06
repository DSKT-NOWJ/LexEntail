import logging

import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
)

from bi_encoder.data import (
    AugmentationType,
    BiEncoderDataset,
    SamplingStrategy,
)
from cross_encoder.data import (
    CrossEncoderDataset,
    GwenBatchCollator,
    MonoT5BatchCollator,
)
from cross_encoder.metrics import compute_metrics_gwen, compute_metrics_monoT5
from cross_encoder.trainer import CrossEncoderTrainer
from settings import (
    ARCHITECTURE,
    AUGMENTATION_TYPE,
    DATASET_PATH,
    DYNAMIC_SAMPLING,
    DYNAMIC_SAMPLING_STRATEGY,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    IS_FP16,
    LEARNING_RATE,
    LOGGING_STEPS,
    LOGGING_STRATEGY,
    LOSS_FUNCTION,
    MAX_NEGATIVES_PER_POSITIVE,
    MODEL_TYPE,
    N_EPOCH,
    OPTIM,
    OUTPUT_DIR,
    PRETRAINED_MODEL,
    TRAIN_BATCH_SIZE,
    TRAINING_SAMPLES_FILE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    YEAR,
)

logging.disable(logging.WARNING)


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    print(f"Using device: {device}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Loss function: {LOSS_FUNCTION}")
    # Initialize wandb

    # Create datasets with architecture specification
    if ARCHITECTURE == "cross_encoder":
        print(
            f"Training {PRETRAINED_MODEL} with params: Dynamic sampling strategy: {DYNAMIC_SAMPLING_STRATEGY}, Max negatives per positive: {MAX_NEGATIVES_PER_POSITIVE}"
        )
        train_dataset = CrossEncoderDataset(
            data_path=DATASET_PATH,
            year=YEAR,
            segment="train",
            architecture=ARCHITECTURE,
            max_neg_per_pos=MAX_NEGATIVES_PER_POSITIVE,
            ns_strategy=DYNAMIC_SAMPLING_STRATEGY,
            training_samples_file=TRAINING_SAMPLES_FILE,
        )

        eval_dataset = CrossEncoderDataset(
            data_path=DATASET_PATH,
            year=YEAR,
            segment="dev",
            architecture=ARCHITECTURE,
        )
    elif ARCHITECTURE == "bi_encoder":
        # Convert string strategy to enum
        sampling_strategy_map = {
            "random": SamplingStrategy.RANDOM,
            "hard": SamplingStrategy.HARD,
            "mixed": SamplingStrategy.MIXED,
        }
        strategy_enum = sampling_strategy_map.get(
            DYNAMIC_SAMPLING_STRATEGY.lower(), SamplingStrategy.RANDOM
        )

        # Convert string augmentation to enum
        augmentation_map = {
            "standard": AugmentationType.STANDARD,
            "augmented": AugmentationType.AUGMENTED,
        }
        augmentation_enum = augmentation_map.get(
            AUGMENTATION_TYPE.lower(), AugmentationType.STANDARD
        )

        train_dataset = BiEncoderDataset(
            {
                "data_path": DATASET_PATH,
                "year": YEAR,
                "segment": "train",
                "architecture": ARCHITECTURE,
                "max_negatives_per_positive": MAX_NEGATIVES_PER_POSITIVE,
                "dynamic_sampling": DYNAMIC_SAMPLING,
                "dynamic_sampling_strategy": strategy_enum,
                "augmentation_type": augmentation_enum,
                "training_samples_file": TRAINING_SAMPLES_FILE,
            }
        ).data

        eval_dataset_obj = BiEncoderDataset(
            {
                "data_path": DATASET_PATH,
                "year": YEAR,
                "segment": "dev",
                "architecture": ARCHITECTURE,
            }
        )
        eval_queries, eval_corpus, eval_relevant_docs = (
            eval_dataset_obj.queries,
            eval_dataset_obj.corpus,
            eval_dataset_obj.relevant_docs,
        )
        eval_dataset = eval_dataset_obj.data

    else:
        raise ValueError(f"Invalid architecture: {ARCHITECTURE}")

    print(f"Training dataset size: {len(train_dataset)} samples")
    print(f"Evaluation dataset size: {len(eval_dataset)} samples")

    # Training arguments

    if ARCHITECTURE == "cross_encoder":
        train_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            save_strategy=LOGGING_STRATEGY,
            save_steps=LOGGING_STEPS,
            logging_strategy=LOGGING_STRATEGY,
            logging_steps=LOGGING_STEPS,
            eval_strategy=LOGGING_STRATEGY,
            eval_steps=LOGGING_STEPS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            num_train_epochs=N_EPOCH,
            warmup_ratio=WARMUP_RATIO,
            optim=OPTIM,
            save_total_limit=3,
            metric_for_best_model="recall",
            fp16=IS_FP16,
            seed=42,
            disable_tqdm=False,
            report_to="none",  # Changed from "none" to "wandb"
            load_best_model_at_end=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        if MODEL_TYPE == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(PRETRAINED_MODEL)
        elif MODEL_TYPE == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
        else:
            raise ValueError(f"Invalid model type: {MODEL_TYPE}")

        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        # Create batch collator
        if any(model_type in PRETRAINED_MODEL for model_type in ["Qwen"]):
            collator_fn = GwenBatchCollator(
                tokenizer=tokenizer,
                device=device,
            )
        elif any(model_type in PRETRAINED_MODEL for model_type in ["monot5"]):
            collator_fn = MonoT5BatchCollator(
                tokenizer=tokenizer,
                device=device,
            )
        else:
            raise ValueError(f"Invalid model type: {PRETRAINED_MODEL}")

        compute_metrics_fn = (
            compute_metrics_monoT5 if MODEL_TYPE == "seq2seq" else compute_metrics_gwen
        )
        # Initialize trainer
        trainer = CrossEncoderTrainer(
            loss_func=LOSS_FUNCTION,
            model_type=MODEL_TYPE,
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator_fn,
            compute_metrics=lambda eval_preds: compute_metrics_fn(
                eval_preds, tokenizer
            ),
        )
    elif ARCHITECTURE == "bi_encoder":
        training_args = SentenceTransformerTrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=N_EPOCH,
            learning_rate=LEARNING_RATE,
            optim=OPTIM,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            fp16=IS_FP16,
            eval_strategy=LOGGING_STRATEGY,
            save_strategy=LOGGING_STRATEGY,
            eval_steps=LOGGING_STEPS,
            save_steps=LOGGING_STEPS,
            save_total_limit=3,
            metric_for_best_model="cosine_recall@10",
            seed=42,
            disable_tqdm=False,
            report_to="none",  # Changed from "none" to "wandb"
            load_best_model_at_end=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        retrieval_evaluator = InformationRetrievalEvaluator(
            queries=eval_queries,
            corpus=eval_corpus,
            relevant_docs=eval_relevant_docs,
            show_progress_bar=True,
            precision_recall_at_k=[1, 3, 5, 10, 20, 50, 100],
            batch_size=EVAL_BATCH_SIZE,
        )

        model = SentenceTransformer(PRETRAINED_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        if LOSS_FUNCTION == "contrastive":
            train_loss = losses.ContrastiveLoss(model=model)
        elif LOSS_FUNCTION == "online_contrastive":
            train_loss = losses.OnlineContrastiveLoss(model=model)
        elif LOSS_FUNCTION == "multiple_negative_ranking":
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
        else:
            raise ValueError(f"Invalid loss function: {LOSS_FUNCTION}")

        # Use custom BiEncoderTrainer for better loss function support
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            evaluator=retrieval_evaluator,
            loss=train_loss,
        )
    else:
        raise ValueError(f"Invalid architecture: {ARCHITECTURE}")

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model
    print("Saving model...")
    trainer.save_model(f"{OUTPUT_DIR}/{PRETRAINED_MODEL.split('/')[-1]}")
    trainer.save_state()

    # # Log final model artifacts to wandb (optional)
    # if os.path.exists(OUTPUT_DIR):
    #     wandb.save(os.path.join(OUTPUT_DIR, "*.json"))  # Save config files
    #     wandb.save(os.path.join(OUTPUT_DIR, "*.bin"))   # Save model weights
    #     wandb.save(os.path.join(OUTPUT_DIR, "*.safetensors"))  # Save safetensors if present

    print("Training completed!")


if __name__ == "__main__":
    train()
