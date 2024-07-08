import logging

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from textpredict.utils.error_handling import ModelError, log_and_raise
from textpredict.utils.evaluation import compute_metrics, log_metrics

logger = logging.getLogger(__name__)


class TextEvaluator:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)

    def tokenize_dataset(self, dataset):
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=True)

        return dataset.map(preprocess_function, batched=True)

    def evaluate(self, eval_data, **kwargs):
        try:
            training_args = TrainingArguments(
                output_dir=kwargs.get("output_dir", "./results"),
                per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 8),
                logging_dir=kwargs.get("logging_dir", "./logs"),
                logging_steps=kwargs.get("logging_steps", 200),
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=self.tokenize_dataset(eval_data),
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
            )

            eval_metrics = trainer.evaluate()
            log_metrics(eval_metrics)
            return eval_metrics
        except Exception as e:
            log_and_raise(ModelError, f"Error during evaluation: {e}")
