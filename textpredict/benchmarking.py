import logging
import os
import time

import psutil
import torch

from textpredict.utils.error_handling import ModelError, log_and_raise

logger = logging.getLogger(__name__)


def measure_inference_time(model, dataset, device="cpu"):
    try:
        model.to(device)  # Move the model to the device
        model.eval()  # Set the model to evaluation mode

        start_time = time.time()
        for batch in dataset:
            inputs = {
                key: val.to(device)
                for key, val in batch.items()
                if key in ["input_ids", "attention_mask"]
            }
            if not inputs:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )
            with torch.no_grad():
                outputs = model(**inputs)
        end_time = time.time()

        inference_time = end_time - start_time
        logger.info(f"Inference time: {inference_time} seconds")
        return inference_time
    except Exception as e:
        log_and_raise(ModelError, f"Error measuring inference time: {e}")


def measure_memory_usage(model, dataset, device="cpu"):
    try:
        model.to(device)
        model.eval()

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss  # in bytes

        for batch in dataset:
            inputs = {
                key: val.to(device)
                for key, val in batch.items()
                if key in ["input_ids", "attention_mask"]
            }
            if not inputs:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )
            with torch.no_grad():
                outputs = model(**inputs)

        mem_after = process.memory_info().rss  # in bytes
        memory_usage = mem_after - mem_before
        logger.info(f"Memory usage: {memory_usage / (1024 ** 2)} MB")  # Convert to MB
        return memory_usage
    except Exception as e:
        log_and_raise(ModelError, f"Error measuring memory usage: {e}")


def benchmark_model(model, dataset, device="cpu"):
    try:
        inference_time = measure_inference_time(model, dataset, device)
        memory_usage = measure_memory_usage(model, dataset, device)
        return {
            "inference_time": inference_time,
            "memory_usage": memory_usage,
        }
    except Exception as e:
        log_and_raise(ModelError, f"Error benchmarking model: {e}")
