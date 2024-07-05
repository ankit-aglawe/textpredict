# textpredict/utils/visualization.py

import itertools
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_metrics(metrics, output_dir):
    """
    Plot training and evaluation metrics.

    Args:
        metrics (dict): Dictionary containing lists of metric values (e.g., {'accuracy': [0.8, 0.85, ...]}).
        output_dir (str): Directory to save the plots.
    """
    try:
        for metric_name, values in metrics.items():
            plt.figure()
            plt.plot(values, label=metric_name)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.capitalize())
            plt.title(f"{metric_name.capitalize()} Over Epochs")
            plt.legend()
            plt.grid(True)
            plot_path = f"{output_dir}/{metric_name}.png"
            plt.savefig(plot_path)
            logger.info(f"{metric_name.capitalize()} plot saved to {plot_path}")
            plt.close()
    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")
        raise


def show_confusion_matrix(confusion_matrix, labels, output_dir):
    """
    Plot and save the confusion matrix.

    Args:
        confusion_matrix (ndarray): Confusion matrix from model predictions.
        labels (list): List of label names.
        output_dir (str): Directory to save the confusion matrix plot.
    """
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = range(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        thresh = confusion_matrix.max() / 2.0
        for i, j in itertools.product(
            range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
        ):
            plt.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plot_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(plot_path)
        logger.info(f"Confusion matrix plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise
