# textpredict/utils/fine_tuning.py

from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from textpredict.logger import get_logger

logger = get_logger(__name__)


def tokenize_and_encode(dataset, tokenizer, max_length=128):
    """
    Tokenize and encode the dataset.

    Args:
        dataset: The dataset to tokenize and encode.
        tokenizer: The tokenizer to use.
        max_length (int, optional): The maximum length of the sequences. Defaults to 128.

    Returns:
        The tokenized and encoded dataset.
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_function, batched=True)


def fine_tune_model(
    model, tokenizer, train_dataset, eval_dataset=None, output_dir="./results", **kwargs
):
    """
    Fine-tune the model with the given training data.

    Args:
        model: The model to be fine-tuned.
        tokenizer: The tokenizer used for encoding the data.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset (optional).
        output_dir (str): The directory to save the fine-tuned model and checkpoints.
        kwargs: Additional keyword arguments for TrainingArguments.
    """
    # Tokenize and encode the datasets
    train_dataset = tokenize_and_encode(train_dataset, tokenizer)
    if eval_dataset:
        eval_dataset = tokenize_and_encode(eval_dataset, tokenizer)

    eval_steps = kwargs.get("eval_steps", 200)
    save_steps = kwargs.get(
        "save_steps", eval_steps * 2
    )  # Ensure save_steps is a multiple of eval_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        logging_dir=f"{output_dir}/logs",
        logging_steps=kwargs.get("logging_steps", 200),
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=kwargs.get("batch_size", 8),
        per_device_eval_batch_size=kwargs.get("eval_batch_size", 8),
        num_train_epochs=kwargs.get("num_train_epochs", 3),
        learning_rate=kwargs.get("learning_rate", 5e-5),
        weight_decay=kwargs.get("weight_decay", 0.01),
        load_best_model_at_end=kwargs.get("load_best_model_at_end", True),
        metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
        save_total_limit=kwargs.get("save_total_limit", 3),
        fp16=kwargs.get("fp16", False),
    )

    callbacks = []
    if "early_stopping_patience" in kwargs:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=kwargs["early_stopping_patience"]
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=callbacks if callbacks else None,
    )

    logger.info("Starting fine-tuning...")
    train_result = trainer.train()
    logger.info("Fine-tuning completed.")

    train_metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    if eval_dataset:
        logger.info("Starting evaluation...")
        eval_metrics = trainer.evaluate()
        logger.info("Evaluation completed.")

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    return trainer
