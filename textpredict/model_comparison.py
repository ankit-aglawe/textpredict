import logging

from textpredict.evaluators.sequence_classification_evaluator import (
    SequenceClassificationEvaluator,
)

logger = logging.getLogger(__name__)


class ModelComparison:
    def __init__(self, models, dataset, task):
        self.models = models
        self.dataset = dataset
        self.task = task
        logger.info(f"ModelComparison initialized for task {task}")

    def compare(self):
        """
        Compare multiple models on the same dataset.

        Returns:
            dict: The comparison results.
        """
        results = {}
        try:
            for model_name in self.models:
                evaluator = SequenceClassificationEvaluator(model_name=model_name)
                evaluator.load_data(dataset_name=self.dataset)
                evaluator.preprocess_data(tokenizer_name=model_name)
                results[model_name] = evaluator.evaluate()
            logger.info("Model comparison completed")
        except Exception as e:
            logger.error(f"Error during model comparison: {e}")
            raise
        return results
