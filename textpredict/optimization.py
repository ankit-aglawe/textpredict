import logging

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from textpredict.utils.error_handling import log_and_raise

logger = logging.getLogger(__name__)


def create_optimizer(model, learning_rate=5e-5):
    try:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        logger.info(f"Optimizer created with learning rate {learning_rate}")
        return optimizer
    except Exception as e:
        log_and_raise(RuntimeError, f"Error creating optimizer: {e}")


def create_scheduler(optimizer, num_training_steps, num_warmup_steps=0):
    try:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        logger.info(
            f"Scheduler created with {num_training_steps} training steps and {num_warmup_steps} warmup steps"
        )
        return scheduler
    except Exception as e:
        log_and_raise(RuntimeError, f"Error creating scheduler: {e}")


def optimize_model(
    model, train_dataloader, eval_dataloader, num_epochs=3, learning_rate=5e-5
):
    try:
        optimizer = create_optimizer(model, learning_rate)
        num_training_steps = len(train_dataloader) * num_epochs
        scheduler = create_scheduler(optimizer, num_training_steps)

        for epoch in range(num_epochs):
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()
            eval_loss /= len(eval_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Evaluation Loss: {eval_loss}")

        logger.info("Model optimization complete")
    except Exception as e:
        log_and_raise(RuntimeError, f"Error optimizing model: {e}")
