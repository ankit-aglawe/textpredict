import itertools
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Visualization:
    def plot_metrics(self, metrics, output_dir="./results"):
        """
        Plot training metrics.

        Args:
            metrics (dict): The metrics to plot.
            output_dir (str, optional): The directory to save the plots. Defaults to "./results".
        """
        try:
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    for sub_metric_name, sub_values in values.items():
                        plt.figure()
                        plt.plot(sub_values, label=f"{metric_name}_{sub_metric_name}")
                        plt.xlabel("Epoch")
                        plt.ylabel(sub_metric_name.capitalize())
                        plt.title(
                            f"{metric_name.capitalize()} {sub_metric_name.capitalize()} Over Epochs"
                        )
                        plt.legend()
                        plt.grid(True)
                        plot_path = f"{output_dir}/{metric_name}_{sub_metric_name}.png"
                        plt.savefig(plot_path)
                        plt.close()
                else:
                    plt.figure()
                    plt.plot(values, label=metric_name)
                    plt.xlabel("Epoch")
                    plt.ylabel(metric_name.capitalize())
                    plt.title(f"{metric_name.capitalize()} Over Epochs")
                    plt.legend()
                    plt.grid(True)
                    plot_path = f"{output_dir}/{metric_name}.png"
                    plt.savefig(plot_path)
                    plt.close()
            logger.info(f"Metrics plots saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")
            raise

    def show_confusion_matrix(self, confusion_matrix, labels, output_dir="./results"):
        """
        Plot the confusion matrix.

        Args:
            confusion_matrix (ndarray): The confusion matrix to plot.
            labels (list): The labels for the confusion matrix.
            output_dir (str, optional): The directory to save the plot. Defaults to "./results".
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
