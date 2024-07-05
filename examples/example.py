import os

from datasets import load_dataset

from textpredict import TextPredict
from textpredict.benchmarking import (
    benchmark_model,
    measure_inference_time,
    measure_memory_usage,
)
from textpredict.config_management import ConfigManager
from textpredict.datasets import get_dataset_splits, load_data
from textpredict.distributed_training import setup_distributed_training
from textpredict.logger import set_logging_level
from textpredict.model_loader import load_model


def test_textpredict():
    tp = TextPredict()
    print(tp.analyse("I love using this package!", task="sentiment"))
    print(tp.analyse("I am excited about this!", task="emotion"))
    print(
        tp.analyse(
            "This package is great for zero-shot learning.",
            task="zeroshot",
            class_list=["positive", "negative", "neutral"],
        )
    )

    # Fine-tuning example (assuming we have a dataset)
    dataset = load_dataset("imdb")
    tp.tune_model(
        task="sentiment",
        training_data=dataset["train"],
        eval_data=dataset["test"],
        num_train_epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        early_stopping_patience=3,
    )

    # Evaluate the fine-tuned model
    metrics = tp.evaluate_model(task="sentiment", eval_data=dataset["test"])
    print("Evaluation metrics:", metrics)

    # Save the fine-tuned model
    tp.save_model(task="sentiment", output_dir="./fine_tuned_sentiment_model")

    # Load the saved model
    tp.load_model(task="sentiment", model_dir="./fine_tuned_sentiment_model")
    print(tp.analyse("I love using this package after fine-tuning!", task="sentiment"))


def test_config_management():
    config_path = "config.json"
    config_manager = ConfigManager(config_path)
    config = config_manager.get("some_key")
    print("Config Value:", config)
    config_manager.set("some_key", "new_value")
    print("Updated Config Value:", config_manager.get("some_key"))


def test_datasets():
    train_data = load_data("imdb", split="train")
    test_data = load_data("imdb", split="test")
    splits = get_dataset_splits("imdb")
    print("Train Data Sample:", train_data[0])
    print("Test Data Sample:", test_data[0])
    print("Dataset Splits:", splits)


def test_distributed_training():
    dataset = load_dataset("imdb")
    setup_distributed_training(
        model="bert-base-uncased",
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        output_dir="./results",
    )


def test_logging():
    set_logging_level("DEBUG")


def test_model_loader():
    model = load_model("distilbert-base-uncased", "sentiment")
    print("Model Loaded:", model)


def test_benchmarking():
    dataset = load_dataset(
        "imdb", split="test[:100]"
    )  # Use a small subset for benchmarking
    model = load_model("distilbert-base-uncased", "sentiment")
    inference_time = measure_inference_time(model, dataset)
    memory_usage = measure_memory_usage(model, dataset)
    benchmark_results = benchmark_model(model, dataset)
    print("Inference Time:", inference_time)
    print("Memory Usage:", memory_usage)
    print("Benchmark Results:", benchmark_results)


def test_cli():
    os.system(
        'poetry run python -m textpredict.cli analyze "I love using this package!" --task sentiment'
    )


def test_web_interface():
    os.system("poetry run uvicorn textpredict.web_interface:app --reload")


if __name__ == "__main__":
    test_textpredict()
    test_config_management()
    test_datasets()
    test_distributed_training()
    test_logging()
    test_model_loader()
    test_benchmarking()
    test_cli()
    test_web_interface()
