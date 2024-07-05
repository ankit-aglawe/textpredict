# textpredict/distributed_training.py

import logging

import torch
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)


def setup_distributed_training(
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
):
    """
    Set up and start distributed training for the given model and datasets.

    Args:
        model: The model to train.
        train_dataset: The dataset to use for training.
        eval_dataset: The dataset to use for evaluation.
        output_dir (str): The directory to save the trained model and checkpoints.
        num_train_epochs (int, optional): The number of epochs to train for. Defaults to 3.
        per_device_train_batch_size (int, optional): The batch size per device during training. Defaults to 16.
        learning_rate (float, optional): The learning rate for training. Defaults to 5e-5.
    """
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to="none",  # Disable reporting to wandb or other platforms by default
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
            dataloader_num_workers=4,
            distributed_type=(
                "multi-node" if torch.cuda.device_count() > 1 else "single-device"
            ),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        logger.info("Starting distributed training")
        trainer.train()

        logger.info("Distributed training complete")
        trainer.save_model(output_dir)
    except Exception as e:
        logger.error(f"Error during distributed training: {e}")
        raise
