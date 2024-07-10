import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

TASK_PIPELINE_MAPPING = {
    "sentiment": "sentiment-analysis",
    "ner": "ner",
    "emotion": "text-classification",
    "zeroshot": "zero-shot-classification",
}


class Explainability:
    def __init__(self, model_name, task, device="cpu"):
        self.model_name = model_name
        self.task = task
        self.device = device
        self.pipeline_task = TASK_PIPELINE_MAPPING.get(task)

        if not self.pipeline_task:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {list(TASK_PIPELINE_MAPPING.keys())}"
            )

        self.pipeline = pipeline(
            self.pipeline_task, model=model_name, device=0 if device == "cuda" else -1
        )
        logger.info(
            f"Explainability initialized with model {model_name} for task {task} on {device}"
        )

    def feature_importance(self, text):
        """
        Get feature importance for a given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: The feature importance.
        """
        try:
            explanation = self.pipeline(text)
            return explanation
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            raise
