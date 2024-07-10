from transformers import AutoModelForSequenceClassification

from .base_evaluator import BaseEvaluator


class SequenceClassificationEvaluator(BaseEvaluator):
    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name)

    def get_detailed_metrics(self):
        """
        Retrieve detailed evaluation metrics specific to sequence classification.
        """
        pass

    def save_results(self, file_path):
        """
        Save evaluation results to a file.

        Args:
            file_path (str): The file path to save the results.
        """
        pass
