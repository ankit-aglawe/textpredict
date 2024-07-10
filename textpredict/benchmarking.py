import logging
import os
import time

import psutil
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from textpredict.evaluation import compute_metrics
from textpredict.model_loader import load_model
from textpredict.utils.error_handling import ModelError, log_and_raise

logger = logging.getLogger(__name__)


class Benchmarking:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = load_model(model_name, task="sentiment")
        self.tokenizer = self.load_tokenizer(model_name)
        logger.info(f"Benchmarking initialized with model {model_name} on {device}")

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def benchmark(self, dataset):
        try:
            training_args = TrainingArguments(output_dir="./results")

            trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
            )
            metrics = trainer.evaluate()
            return metrics
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            raise

    def measure_inference_time(self, dataset):
        try:
            self.model.to(self.device)
            self.model.eval()
            start_time = time.time()
            for batch in dataset:
                inputs = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key in ["input_ids", "attention_mask"]
                }
                if not inputs:
                    raise ValueError(
                        "You have to specify either input_ids or inputs_embeds"
                    )
                with torch.no_grad():
                    outputs = self.model(**inputs)
            end_time = time.time()
            inference_time = end_time - start_time
            logger.info(f"Inference time: {inference_time} seconds")
            return inference_time
        except Exception as e:
            log_and_raise(ModelError, f"Error measuring inference time: {e}")

    def measure_memory_usage(self, dataset):
        try:
            self.model.to(self.device)
            self.model.eval()
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss  # in bytes
            for batch in dataset:
                inputs = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key in ["input_ids", "attention_mask"]
                }
                if not inputs:
                    raise ValueError(
                        "You have to specify either input_ids or inputs_embeds"
                    )
                with torch.no_grad():
                    outputs = self.model(**inputs)
            mem_after = process.memory_info().rss  # in bytes
            memory_usage = mem_after - mem_before
            logger.info(
                f"Memory usage: {memory_usage / (1024 ** 2)} MB"
            )  # Convert to MB
            return memory_usage
        except Exception as e:
            log_and_raise(ModelError, f"Error measuring memory usage: {e}")
