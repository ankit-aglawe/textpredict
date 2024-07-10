import logging

from transformers import AutoModelForSeq2SeqLM

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Seq2seqTrainer(BaseTrainer):
    def __init__(self, model_name, output_dir, config=None, device="cpu"):
        super().__init__(model_name, output_dir, config, device)
        logger.info(f"Initialized Seq2SeqTrainer with model {model_name} on {device}")

    def load_model(self, model_name):
        """
        Load the model for sequence-to-sequence tasks.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            AutoModelForSeq2SeqLM: The loaded model.
        """
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
