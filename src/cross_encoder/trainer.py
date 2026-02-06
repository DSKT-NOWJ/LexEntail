import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import Trainer

# Add framework root to path for imports
root = Path(__file__).parents[2]  # Go up to framework root
sys.path.insert(0, str(root))

from src.settings import W_LOSS


class CrossEncoderTrainer(Trainer):
    def __init__(self, loss_func: str, model_type: str, *args, **kwargs):
        """
        Args:
            loss_func (str): loss function
            model_type (str): model type
        """
        super().__init__(*args, **kwargs)

        self.decoder_input_ids = None
        self.loss_func = loss_func
        self.model_type = model_type

        if self.model_type == "seq2seq":
            self.token_false_id = self.tokenizer.get_vocab()["▁false"]
            self.token_true_id = self.tokenizer.get_vocab()["▁true"]

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        outputs = None

        if self.decoder_input_ids is None:
            if self.model_type == "seq2seq":
                if isinstance(model, torch.nn.DataParallel) or isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ):
                    self.decoder_input_ids = model.module._shift_right(inputs["labels"])
                else:
                    self.decoder_input_ids = model._shift_right(inputs["labels"])

                inputs["decoder_input_ids"] = self.decoder_input_ids

        if self.loss_func == "cross_entropy":
            outputs = model(**inputs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss
            else:
                return super().compute_loss(
                    model, inputs, num_items_in_batch, return_outputs
                )

        elif self.loss_func == "contrastive":
            outputs = model(**inputs, use_cache=False)
            xe_loss, logits = outputs[:2]
            logits = logits[:, -1, [self.token_false_id, self.token_true_id]]
            scores = torch.nn.functional.log_softmax(logits, dim=1)
            log_probs = scores[:, 1]
            loss = torch.mean(
                -torch.log(
                    torch.exp(log_probs[0]) / torch.sum(torch.exp(log_probs), dim=-1)
                )
            )

        elif self.loss_func == "weighted_cross_entropy":
            xe_loss, logits = model(**inputs, use_cache=False)[:2]
            # For T5 models, we need to create a simple weighted loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            loss = W_LOSS * loss_fct(ignore_index=-100, reduction="none")(
                logits.view(-1, logits.size(-1)), inputs["labels"].view(-1)
            )
            # Apply weighting if needed
            loss = torch.mean(loss)
        else:
            raise ValueError(self.loss_func)

        return (loss, outputs) if return_outputs else loss
