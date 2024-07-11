from transformers import AutoModelForTokenClassification

from textpredict.logger import get_logger

from .base_evaluator import BaseEvaluator

logger = get_logger(__name__)


class TokenClassificationEvaluator(BaseEvaluator):
    def load_model(self, model_name, device):
        device = device

        try:
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            model.to(device)
            return model

        except Exception as e:

            logger.warnings(f"Failed to load model on {device}: {e}")
            logger.info("Falling back to CPU.")
            self.device = "cpu"
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            model.to(device)

            logger.info(
                f"Initialized Seq2SeqTrainer with model {model_name} on {device}"
            )

            return model

    def get_detailed_metrics(self):
        """
        Retrieve detailed evaluation metrics specific to token classification.
        """
        pass

    def save_results(self, file_path):
        """
        Save evaluation results to a file.

        Args:
            file_path (str): The file path to save the results.
        """
        pass
