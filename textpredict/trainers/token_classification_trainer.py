import logging

from transformers import AutoModelForTokenClassification

from textpredict.evaluators import TokenClassificationEvaluator

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TokenClassificationTrainer(BaseTrainer):
    def __init__(self, model_name, output_dir, config=None, device="cpu"):
        super().__init__(model_name, output_dir, config, device)
        logger.info(
            f"Initialized TokenClassificationTrainer with model {model_name} on {device}"
        )

    def load_model(self, model_name):
        """
        Load the model for token classification.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            AutoModelForTokenClassification: The loaded model.
        """
        return AutoModelForTokenClassification.from_pretrained(model_name)

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

        evaluator = TokenClassificationEvaluator(
            model_name=self.output_dir,
            device=self.device,
            evaluation_config=evaluation_config,
        )

        evaluator.data = test_dataset
        eval_metrics = evaluator.evaluate()
        return eval_metrics
