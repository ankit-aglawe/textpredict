import logging

from datasets import load_dataset  # type: ignore
from textpredict import Seq2seqTrainer, initialize, load_data, set_device

logging.basicConfig(level=logging.INFO)


def train_seq2seq():
    set_device("cpu")

    # Load the dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    # Load and preprocess dataset with specified text column
    dataset = load_data(
        dataset=ds,
        splits=["train", "test"],
        text_column="prompt",
        label_column="code",
    )

    # Initialize the trainer
    trainer = Seq2seqTrainer(
        model_name="google/flan-t5-small",
        output_dir="./seq2seq_model",
        training_config={
            "num_train_epochs": 0.064,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "learning_rate": 3e-5,
            "logging_dir": "./logs",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
        },
        device="cpu",
    )

    # Set datasets
    trainer.train_dataset = dataset["train"]
    trainer.val_dataset = dataset["test"]

    # Start training
    trainer.train()

    # Save the model
    trainer.save()

    # Get training metrics
    metrics = trainer.get_metrics()
    print(f"Training Metrics: {metrics}")

    # Evaluate the model
    evaluate = trainer.evaluate(test_dataset=dataset["test"])
    print(f"Evaluation Metrics: {evaluate}")

    # Load the trained model
    model = initialize(model_name="./seq2seq_model", task="seq2seq")

    # Analyze a sample text
    text = "Summarize the following document: ..."
    result = model.analyze(text, return_probs=True)
    print("Result:", result)


if __name__ == "__main__":
    train_seq2seq()
