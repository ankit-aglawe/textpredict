from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from textpredict.logger import get_logger
from textpredict.utils.error_handling import ModelError, log_and_raise

logger = get_logger(__name__)


def tokenize_dataset(dataset, tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    return dataset.map(preprocess_function, batched=True)


def setup_distributed_training(
    model_name, train_dataset, eval_dataset, output_dir, **kwargs
):
    """
    Setup distributed training for a model using the provided datasets.

    Args:
        model_name (str): The name of the model to train.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        output_dir (str): The directory to save training results.
        **kwargs: Additional keyword arguments for TrainingArguments.

    Returns:
        None
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize the datasets
        train_dataset = tokenize_dataset(train_dataset, tokenizer)
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 8),
            num_train_epochs=kwargs.get("num_train_epochs", 3),
            evaluation_strategy=kwargs.get("evaluation_strategy", "steps"),
            save_steps=kwargs.get("save_steps", 10_000),
            save_total_limit=kwargs.get("save_total_limit", 2),
            learning_rate=kwargs.get("learning_rate", 5e-5),
            logging_dir=kwargs.get("logging_dir", "./logs"),
            logging_steps=kwargs.get("logging_steps", 500),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
    except Exception as e:
        log_and_raise(ModelError, f"Error during distributed training: {e}")
