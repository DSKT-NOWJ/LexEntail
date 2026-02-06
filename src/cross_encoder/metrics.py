import numpy as np
from transformers import EvalPrediction

def compute_metrics_monoT5(eval_preds: EvalPrediction, tokenizer):
    preds, labels = eval_preds
    logits, _ = preds

    token_true_id = tokenizer.get_vocab()["▁true"]
    token_false_id = tokenizer.get_vocab()["▁false"]
    
    last_logits = logits[:, -1, :]
    true_logits = last_logits[:, token_true_id]
    false_logits = last_logits[:, token_false_id]
    
    # Replace -100 in labels with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    predictions = (true_logits > false_logits).astype(int)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [1 if label == "true" else 0 for label in decoded_labels]
    
    tp = np.sum([1 if a == b and a == 1 else 0 for a, b in zip(predictions, decoded_labels)])
    fp = np.sum([1 if a != b and a == 0 else 0 for a, b in zip(predictions, decoded_labels)])
    fn = np.sum([1 if a != b and a == 1 else 0 for a, b in zip(predictions, decoded_labels)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall' : recall,
        'f1': f1,
    }
    
def compute_metrics_gwen(eval_preds: EvalPrediction, tokenizer):
    logits, labels = eval_preds
    
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    
    last_logits = logits[:, -1, :]
    true_logits = last_logits[:, token_true_id]
    false_logits = last_logits[:, token_false_id]
    
    # Replace -100 in labels with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    predictions = (true_logits > false_logits).astype(int)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [1 if label == "yes" else 0 for label in decoded_labels]
    
    tp = np.sum([1 if a == b and a == 1 else 0 for a, b in zip(predictions, decoded_labels)])
    fp = np.sum([1 if a != b and a == 0 else 0 for a, b in zip(predictions, decoded_labels)])
    fn = np.sum([1 if a != b and a == 1 else 0 for a, b in zip(predictions, decoded_labels)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall' : recall,
        'f1': f1,
    }