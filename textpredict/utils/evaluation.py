# textpredict/utils/evaluation.py

import logging

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_metrics(p: EvalPrediction):
    """
    Compute metrics for evaluation.

    Args:
        p (EvalPrediction): The evaluation predictions and label_ids.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
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
    Log the metrics.

    Args:
        metrics (list or dict): The metrics to log. Can be a list of metric dicts or a single metric dict.
    """
    if isinstance(metrics, list):
        for metric_dict in metrics:
            for metric, value in metric_dict.items():
                logger.info(f"{metric} = {value}")
    elif isinstance(metrics, dict):
        for metric, value in metrics.items():
            logger.info(f"{metric} = {value}")
    else:
        logger.error("Metrics format not recognized. Expected list or dict.")
