# textpredict/benchmarking.py

import logging
import time

import torch

logger = logging.getLogger(__name__)


def measure_inference_time(model, dataloader, device="cuda"):
    """
    Measure the average inference time of the model.

    Args:
        model: The model to benchmark.
        dataloader: The data loader with input data.
        device (str, optional): The device to run the inference on. Defaults to 'cuda'.

    Returns:
        float: The average inference time per sample in milliseconds.
    """
    try:
        model.to(device)
        model.eval()

        start_time = time.time()
        num_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: value.to(device) for key, value in batch.items()}
                _ = model(**inputs)
                num_samples += len(inputs["input_ids"])

        total_time = time.time() - start_time
        avg_time_per_sample = (
            total_time / num_samples
        ) * 1000  # Convert to milliseconds
        logger.info(f"Average inference time per sample: {avg_time_per_sample:.2f} ms")
        return avg_time_per_sample
    except Exception as e:
        logger.error(f"Error measuring inference time: {e}")
        raise


def measure_memory_usage(model, dataloader, device="cuda"):
    """
    Measure the peak memory usage during inference.

    Args:
        model: The model to benchmark.
        dataloader: The data loader with input data.
        device (str, optional): The device to run the inference on. Defaults to 'cuda'.

    Returns:
        float: The peak memory usage in MB.
    """
    try:
        model.to(device)
        model.eval()

        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: value.to(device) for key, value in batch.items()}
                _ = model(**inputs)

        peak_memory = torch.cuda.max_memory_allocated(device) / (
            1024**2
        )  # Convert to MB
        logger.info(f"Peak memory usage: {peak_memory:.2f} MB")
        return peak_memory
    except Exception as e:
        logger.error(f"Error measuring memory usage: {e}")
        raise


def benchmark_model(model, dataloader, device="cuda"):
    """
    Benchmark the model for both inference time and memory usage.

    Args:
        model: The model to benchmark.
        dataloader: The data loader with input data.
        device (str, optional): The device to run the inference on. Defaults to 'cuda'.

    Returns:
        dict: A dictionary containing the average inference time and peak memory usage.
    """
    try:
        avg_inference_time = measure_inference_time(model, dataloader, device)
        peak_memory_usage = measure_memory_usage(model, dataloader, device)
        benchmark_results = {
            "average_inference_time_ms": avg_inference_time,
            "peak_memory_usage_mb": peak_memory_usage,
        }
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results
    except Exception as e:
        logger.error(f"Error benchmarking model: {e}")
        raise
