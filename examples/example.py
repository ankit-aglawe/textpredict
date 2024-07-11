from datasets import load_dataset
from textpredict import (
    Benchmarking,
    Explainability,
    ModelComparison,
    SequenceClassificationEvaluator,
    SequenceClassificationTrainer,
    Visualization,
    clean_text,
    initialize,
    load_data,
    set_device,
)


# Function to test simple prediction using default model
def text_simple_prediction():
    set_device("gpu")

    # sentiment
    texts = ["I love this product!", "I love this product!"]
    text = "I love this product!"
    model = initialize(task="sentiment", device="gpu")
    result = model.analyze(texts, return_probs=False)
    print(f"Simple Prediction Result: {result}")

    # # emotion
    text = ["I am happy today", "I am happy today"]
    model = initialize(task="emotion", device="gpu")
    result = model.analyze(text, return_probs=False)
    print(f"Emotion found : {result}")

    # zeroshot
    texts = ["I am happy today", "I am happy today"]
    text = "I am happy today"
    model = initialize(task="zeroshot", device="gpu")

    result = model.analyze(
        text, candidate_labels=["negative", "positive"], return_probs=True
    )
    print(f"Zeroshot Prediction Result: {result}")

    # ner
    texts = ["I am in London, united kingdom", "I am in Manchester, united kingdom"]
    text = "I am in Manchester, united kingdom"

    model = initialize(task="ner")
    result = model.analyze(text, return_probs=True)
    print(f"NER found : {result}")


# Function to test prediction using a Hugging Face model
def text_hf_prediction():
    set_device("cuda")
    text = "I love this product!"

    model = initialize(
        task="sentiment",
        device="cpu",
        model_name="AnkitAI/reviews-roberta-base-sentiment-analysis",
        source="huggingface",
    )

    result = model.analyze(text, return_probs=True)

    print(f"Hugging Face Prediction Result: {result}")

    # # emotion
    text = ["I am happy today", "I am happy today"]
    model = initialize(
        task="emotion", model_name="AnkitAI/deberta-v3-small-base-emotions-classifier"
    )
    result = model.analyze(text, return_probs=False)
    print(f"Emotion found : {result}")

    # zeroshot
    texts = ["I am happy today", "I am happy today"]
    text = "I am happy today"
    model = initialize(task="zeroshot", source="huggingface")

    result = model.analyze(
        text, candidate_labels=["negative", "positive"], return_probs=True
    )
    print(f"Zeroshot Prediction Result: {result}")

    # ner
    texts = [
        "I am in London, united kingdom",
        "I am in Manchester, united kingdom",
    ]  # noqa: F841
    text = "I am in Manchester, united kingdom"

    model = initialize(task="ner", source="huggingface")
    result = model.analyze(text, return_probs=True)
    print(f"NER found : {result}")


# Function to train a sequence classification model
def train_sequence_classification():
    set_device("cuda")

    # Load and preprocess the dataset
    raw_train_dataset = load_dataset("imdb", split="train[:100]")
    raw_validation_dataset = load_dataset("imdb", split="test[:100]")

    tokenized_train_dataset = load_data(dataset=raw_train_dataset, splits=["train"])
    tokenized_validation_dataset = load_data(
        dataset=raw_validation_dataset, splits=["test"]
    )

    train_dataset = tokenized_train_dataset["train"]
    val_dataset = tokenized_validation_dataset["test"]

    training_config = {
        "num_train_epochs": 0.064,
        "per_device_train_batch_size": 2,
    }

    trainer = SequenceClassificationTrainer(
        model_name="bert-base-uncased",
        output_dir="./results_new",
        device="cuda",
        training_config=training_config,
    )

    # Assign the preprocessed training data to the trainer
    trainer.train_dataset = train_dataset
    trainer.val_dataset = val_dataset  # Set the validation dataset

    # Train the model
    trainer.train(from_checkpoint=True)
    trainer.save()
    metrics = trainer.get_metrics()
    print(f"Training Metrics: {metrics}")

    evaluate = trainer.evaluate(test_dataset=val_dataset)
    print(f"Evaluation Metrics: {evaluate}")

    model = initialize(model_name="./results_new", task="sequence_classification")

    text = "its a good product"

    result = model.analyze(text, return_probs=True)

    print("result", result)


def train_seq2seq():
    from datasets import load_dataset  # type: ignore
    from textpredict import Seq2seqTrainer, load_data

    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    # Load dataset
    dataset = load_data(
        dataset=ds,
        splits=["train", "validation", "test"],
        text_column="prompt",
        label_column="code",
    )

    # Initialize the trainer
    trainer = Seq2seqTrainer(
        model_name="google/flan-t5-small",
        output_dir="./seq2seq_model",
        training_config={
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "learning_rate": 3e-5,
            "logging_dir": "./logs",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
        },
    )

    # Set datasets
    trainer.train_dataset = dataset["train"]
    trainer.val_dataset = dataset["validation"]

    # Start training
    trainer.train()

    # Save the model
    trainer.save()

    metrics = trainer.get_metrics()
    print(f"Training Metrics: {metrics}")

    evaluate = trainer.evaluate(test_dataset=dataset["test"])
    print(f"Evaluation Metrics: {evaluate}")

    model = initialize(model_name="./results_seq2seq", task="seq2seq")

    text = "Summarize the following document: ..."

    result = model.analyze(text, return_probs=True)

    print("result", result)


# def train_token_classification():

#     import torch  # type: ignore
#     from textpredict import TokenClassificationTrainer  # noqa: E402
#     from transformers import AutoTokenizer  # type: ignore

#     # Set device to cuda if available
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#     # Load and preprocess the dataset
#     raw_train_dataset = load_dataset("conll2003", split="train[:100]")
#     raw_validation_dataset = load_dataset("conll2003", split="validation[:100]")

#     # Tokenize the datasets
#     def tokenize_and_align_labels(examples):
#         tokenized_inputs = tokenizer(
#             examples["tokens"],
#             truncation=True,
#             is_split_into_words=True,
#             padding="max_length",
#             max_length=128,
#         )
#         labels = []
#         for i, label in enumerate(examples["ner_tags"]):
#             word_ids = tokenized_inputs.word_ids(batch_index=i)
#             label_ids = []
#             previous_word_idx = None
#             for word_idx in word_ids:
#                 if word_idx is None:
#                     label_ids.append(-100)
#                 elif word_idx != previous_word_idx:
#                     label_ids.append(label[word_idx])
#                 else:
#                     label_ids.append(-100)
#                 previous_word_idx = word_idx
#             labels.append(label_ids)
#         tokenized_inputs["labels"] = labels
#         return tokenized_inputs

#     tokenized_train_dataset = raw_train_dataset.map(
#         tokenize_and_align_labels, batched=True
#     )
#     tokenized_validation_dataset = raw_validation_dataset.map(
#         tokenize_and_align_labels, batched=True
#     )

#     # Set the format for PyTorch tensors
#     tokenized_train_dataset.set_format(
#         type="torch", columns=["input_ids", "attention_mask", "labels"]
#     )
#     tokenized_validation_dataset.set_format(
#         type="torch", columns=["input_ids", "attention_mask", "labels"]
#     )

#     # Define training configuration
#     training_config = {
#         "num_train_epochs": 3,
#         "per_device_train_batch_size": 8,
#     }

#     # Initialize the trainer
#     trainer = TokenClassificationTrainer(
#         model_name="bert-base-uncased",
#         output_dir="./results_token_classification",
#         device=device,
#         training_config=training_config,
#     )

#     # Assign the preprocessed training data to the trainer
#     trainer.train_dataset = tokenized_train_dataset
#     trainer.val_dataset = tokenized_validation_dataset

#     # Train the model
#     trainer.train(from_checkpoint=False)
#     trainer.save()
#     metrics = trainer.get_metrics()
#     print(f"Training Metrics: {metrics}")

#     evaluate = trainer.evaluate(test_dataset=tokenized_validation_dataset)
#     print(f"Evaluation Metrics: {evaluate}")

#     model = initialize(
#         model_name="./results_token_classification", task="token_classification"
#     )

#     text = "Hawking was a theoretical physicist."

#     result = model.analyze(text, return_probs=True)

#     print("result", result)


# Function to evaluate a sequence classification model
def evaluate_sequence_classification():
    set_device("cuda")
    # Load and preprocess the dataset
    raw_test_dataset = load_dataset("imdb", split="test[:10]")

    tokenized_test_dataset = load_data(dataset=raw_test_dataset, splits=["test"])
    test_dataset = tokenized_test_dataset["test"]

    evaluation_config = {
        "per_device_eval_batch_size": 2,
    }

    evaluator = SequenceClassificationEvaluator(
        model_name="bert-base-uncased",
        device="cuda",
        evaluation_config=evaluation_config,
    )

    # Assign the preprocessed test data to the evaluator
    evaluator.data = test_dataset

    # Evaluate the model
    eval_metrics = evaluator.evaluate()
    print(f"Evaluation Metrics: {eval_metrics}")


# Function to benchmark a model
def benchmark_model():
    benchmarker = Benchmarking(model_name="bert-base-uncased", device="cpu")
    dataset = load_dataset("imdb", split="test[:10]")
    dataset = dataset.map(
        lambda x: benchmarker.tokenizer(
            x["text"], padding="max_length", truncation=True
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    performance = benchmarker.benchmark(dataset)
    print(f"Benchmarking Performance: {performance}")


# Function to visualize metrics
def visualize_metrics():
    metrics = {"accuracy": [0.8, 0.85, 0.9], "loss": [0.6, 0.4, 0.2]}
    viz = Visualization()
    viz.plot_metrics(metrics)


# Function to compare models
def compare_models():
    raw_test_dataset = load_dataset("imdb", split="test[:10]")

    tokenized_test_dataset = load_data(dataset=raw_test_dataset, splits=["test"])
    test_dataset = tokenized_test_dataset["test"]

    comparison = ModelComparison(
        models=["bert-base-uncased", "roberta-base"],
        dataset=test_dataset,
        task="sentiment",
    )
    results = comparison.compare()
    print(f"Model Comparison Results: {results}")


# Function to explain a prediction
def explain_prediction():
    text = "I love this product!"
    explainer = Explainability(
        model_name="bert-base-uncased", task="sentiment", device="cpu"
    )
    importance = explainer.feature_importance(text=text)
    print(f"Feature Importance: {importance}")


# Function to clean text using utility functions
def clean_text_example():
    raw_text = "This is some raw text! Check http://example.com"
    cleaned_text = clean_text(raw_text)
    print(f"Cleaned Text: {cleaned_text}")


# Main function to run all tests
def main():
    # print("Running Simple Prediction...")
    # text_simple_prediction()

    # print("\nRunning Hugging Face Prediction...")
    # text_hf_prediction()

    # print("\nTraining Sequence Classification Model...")
    # train_sequence_classification()

    # Run the training function
    print("\Trainig Seq2seq Model...")
    train_seq2seq()

    # Run the training function
    # print("\Trainig Toekn c;assification Model...")
    # train_token_classification()

    # print("\nEvaluating Sequence Classification Model...")
    # evaluate_sequence_classification()

    # print("\nBenchmarking Model...")
    # benchmark_model()

    # print("\nVisualizing Metrics...")
    # visualize_metrics()

    # print("\nComparing Models...")
    # compare_models()

    # print("\nExplaining Prediction...")
    # explain_prediction()

    # print("\nCleaning Text Example...")
    # clean_text_example()


if __name__ == "__main__":
    main()
