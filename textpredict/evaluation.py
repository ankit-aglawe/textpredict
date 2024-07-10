# evaluation.py
import logging

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_metrics(p: EvalPrediction):
    preds = (
        p.predictions.argmax(-1)
        if isinstance(p.predictions, tuple)
        else p.predictions.argmax(-1)
    )
    labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def log_metrics(metrics):
    """
    Log the metrics to the console.

    Args:
        metrics (list or dict): The metrics to log.
    """
    if isinstance(metrics, list):
        for entry in metrics:
            if isinstance(entry, dict):
                for metric, value in entry.items():
                    print(f"{metric}: {value}")
            else:
                print(entry)
    elif isinstance(metrics, dict):
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    else:
        print(metrics)
