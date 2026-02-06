from transformers import EvalPrediction
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

def compute_metrics(eval_prediction: EvalPrediction):
    anchor_embeddings, doc_embeddings = eval_prediction.predictions
    labels = eval_prediction.label_ids

    cosine_sim = 1 - paired_cosine_distances(anchor_embeddings, doc_embeddings)
    pearson, _ = pearsonr(labels, cosine_sim)
    spearman, _ = spearmanr(labels, cosine_sim)
    return {"pearson": round(pearson, 3), "spearman": round(spearman, 3)}