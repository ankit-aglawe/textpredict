from transformers import AutoModelForSequenceClassification

from textpredict import logger
from textpredict.evaluators import SequenceClassificationEvaluator

from .base_trainer import BaseTrainer


class SequenceClassificationTrainer(BaseTrainer):
    def load_model(self, model_name, device):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)
            return model

        except Exception as e:

            logger.warnings(f"Failed to load model on {self.device}: {e}")
            logger.info("Falling back to CPU.")
            self.device = "cpu"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.to(device)

            logger.info(
                f"Initialized Seq2SeqTrainer with model {model_name} on {device}"
            )

            return model

    def evaluate(self, test_dataset, evaluation_config=None):
        """
        Evaluate the trained model using the provided test dataset.

        Args:
            test_dataset (Dataset): The dataset to evaluate the model on.
            evaluation_config (dict, optional): Configuration for evaluation. Defaults to None.

        Returns:
            dict: The evaluation metrics.
        """
        if evaluation_config is None:
            evaluation_config = {}

        evaluator = SequenceClassificationEvaluator(
            model_name=self.output_dir,
            device=self.device,
            evaluation_config=evaluation_config,
        )

        evaluator.data = test_dataset
        eval_metrics = evaluator.evaluate()
        return eval_metrics
