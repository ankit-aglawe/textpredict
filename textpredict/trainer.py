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


class TextTrainer:
    def __init__(self, model_name, output_dir="./results", device="cpu"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)

    def tokenize_dataset(self, dataset):
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding=True)

        return dataset.map(preprocess_function, batched=True)

    def fine_tune(self, training_data, eval_data=None, **kwargs):
        try:
            evaluation_strategy = kwargs.get("evaluation_strategy", "epoch")
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=kwargs.get("num_train_epochs", 3),
                per_device_train_batch_size=kwargs.get(
                    "per_device_train_batch_size", 8
                ),
                per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 8),
                warmup_steps=kwargs.get("warmup_steps", 500),
                weight_decay=kwargs.get("weight_decay", 0.01),
                logging_dir=kwargs.get("logging_dir", "./logs"),
                logging_steps=kwargs.get("logging_steps", 100),
                evaluation_strategy=evaluation_strategy,
                save_strategy=evaluation_strategy,  # Ensure strategies match
                save_total_limit=kwargs.get("save_total_limit", 1),
                load_best_model_at_end=True,
                metric_for_best_model=kwargs.get("metric_for_best_model", "accuracy"),
                greater_is_better=kwargs.get("greater_is_better", True),
                gradient_accumulation_steps=kwargs.get(
                    "gradient_accumulation_steps", 2
                ),
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.tokenize_dataset(training_data),
                eval_dataset=self.tokenize_dataset(eval_data) if eval_data else None,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            log_metrics(trainer.state.log_history)
        except Exception as e:
            log_and_raise(ModelError, f"Error during fine-tuning: {e}")

    def save_model(self):
        try:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Model saved to {self.output_dir}")
        except Exception as e:
            log_and_raise(ModelError, f"Error saving model: {e}")

    def load_model(self, model_dir):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            log_and_raise(ModelError, f"Error loading model: {e}")
