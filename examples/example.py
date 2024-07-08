import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from textpredict import TextPredict
from textpredict.benchmarking import benchmark_model
from textpredict.evaluator import TextEvaluator
from textpredict.trainer import TextTrainer


def test_textpredict():
    tp = TextPredict(device="cpu")

    # Analyse using default models
    result_sentiment = tp.analyse("I love using this package!", task="sentiment")
    print(result_sentiment)

    result_emotion = tp.analyse("I am excited about this!", task="emotion")
    print(result_emotion)

    result_zeroshot = tp.analyse(
        "This package is great for zero-shot learning.",
        task="zeroshot",
        class_list=["positive", "negative", "neutral"],
    )
    print(result_zeroshot)


def test_texttrainer():
    trainer = TextTrainer(
        model_name="bert-base-uncased", output_dir="./fine_tuned_sentiment_model"
    )
    dataset = load_dataset("imdb")

    # Reduce the subset size for testing
    train_pos = dataset["train"].filter(lambda x: x["label"] == 1).select(range(5))
    train_neg = dataset["train"].filter(lambda x: x["label"] == 0).select(range(5))
    train_subset = concatenate_datasets([train_pos, train_neg])

    test_pos = dataset["test"].filter(lambda x: x["label"] == 1).select(range(5))
    test_neg = dataset["test"].filter(lambda x: x["label"] == 0).select(range(5))
    test_subset = concatenate_datasets([test_pos, test_neg])

    trainer.fine_tune(
        training_data=train_subset,
        eval_data=test_subset,
        num_train_epochs=0.064,
        per_device_train_batch_size=2,  # Reduce batch size
        per_device_eval_batch_size=2,  # Reduce batch size
        learning_rate=2e-5,
        early_stopping_patience=3,
    )
    trainer.save_model()


def test_textevaluator():
    evaluator = TextEvaluator(model_name="bert-base-uncased")
    dataset = load_dataset("imdb")
    eval_dataset = dataset["test"].select(range(100))

    eval_metrics = evaluator.evaluate(eval_dataset)
    print(eval_metrics)


def test_benchmarking():
    dataset = load_dataset(
        "imdb", split="test[:100]"
    )  # Use a small subset for benchmarking

    # Create a DataLoader to properly batch the dataset
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    evaluator = TextEvaluator(model_name="distilbert-base-uncased")
    benchmark_results = benchmark_model(evaluator.model, dataloader)
    print("Benchmark Results:", benchmark_results)


if __name__ == "__main__":
    test_textpredict()
    test_texttrainer()
    test_textevaluator()
    test_benchmarking()
