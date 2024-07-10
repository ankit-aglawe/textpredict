import logging

from transformers import AutoModelForSeq2SeqLM

from textpredict.evaluators import Seq2seqEvaluator

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

        evaluator = Seq2seqEvaluator(
            model_name=self.output_dir,
            device=self.device,
            evaluation_config=evaluation_config,
        )

        evaluator.data = test_dataset
        eval_metrics = evaluator.evaluate()
        return eval_metrics
