from transformers import AutoModelForSequenceClassification

from textpredict.evaluators import SequenceClassificationEvaluator

from .base_trainer import BaseTrainer


class SequenceClassificationTrainer(BaseTrainer):
    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name)

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
