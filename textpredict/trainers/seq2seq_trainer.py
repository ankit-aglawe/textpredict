from transformers import AutoModelForSeq2SeqLM

from textpredict import logger
from textpredict.evaluators import Seq2seqEvaluator

from .base_trainer import BaseTrainer


class Seq2seqTrainer(BaseTrainer):
    def __init__(self, model_name, output_dir, config=None, device=None):
        super().__init__(model_name, output_dir, config, device)

    def load_model(self, model_name, device):
        """
        Load the model for sequence-to-sequence tasks.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            AutoModelForSeq2SeqLM: The loaded model.
        """
        device = device

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to(device)

            logger.info(
                f"Initialized Seq2SeqTrainer with model {model_name} on {device}"
            )

            return model

        except Exception as e:

            logger.warning(f"Failed to load model on {self.device}: {e}")
            logger.info("Falling back to CPU.")
            self.device = "cpu"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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

        evaluator = Seq2seqEvaluator(
            model_name=self.output_dir,
            device=self.device,
            evaluation_config=evaluation_config,
        )

        evaluator.data = test_dataset
        eval_metrics = evaluator.evaluate()
        return eval_metrics
