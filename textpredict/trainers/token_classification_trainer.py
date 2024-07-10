import logging

from transformers import AutoModelForTokenClassification

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
