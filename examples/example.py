from datasets import load_dataset

import textpredict as tp


# Function to test simple prediction using default model
def text_simple_prediction():
    # sentiment
    text = ["I love this product!", "I love this product!"]
    model = tp.initialize(task="sentiment")
    result = model.analyze(text, return_probs=True)
    print(f"Simple Prediction Result: {result}")

    # emotion
    text = ["I am happy today", "I am happy today"]
    model = tp.initialize(task="emotion")
    result = model.analyze(text, return_probs=True)
    print(f"Emotion found : {result}")

    # zeroshot
    text = ["I am happy today", "I am happy today"]
    model = tp.initialize(task="zeroshot")

    result = model.analyze(
        text, candidate_labels=["negative", "positive"], return_probs=True
    )
    print(f"Zeroshot Prediction Result: {result}")

    # ner
    text = ["I am in London, united kingdom", "I am in Manchester, united kingdom"]
    model = tp.initialize(task="ner")
    result = model.analyze(text)
    print(f"NER found : {result}")


# Function to test prediction using a Hugging Face model
def text_hf_prediction():
    text = "I love this product!"
    model = tp.initialize(
        task="sentiment",
        device="cpu",
        model_name="bert-base-uncased",
        source="huggingface",
    )
    result = model.analyze(text)
    print(f"Hugging Face Prediction Result: {result}")


# Function to train a sequence classification model
def train_sequence_classification():
    # Load and preprocess the dataset
    raw_train_dataset = load_dataset("imdb", split="train[:10]")
    raw_validation_dataset = load_dataset("imdb", split="test[:10]")

    tokenized_train_dataset = tp.load_data(dataset=raw_train_dataset, splits=["train"])
    tokenized_validation_dataset = tp.load_data(
        dataset=raw_validation_dataset, splits=["test"]
    )

    train_dataset = tokenized_train_dataset["train"]
    val_dataset = tokenized_validation_dataset["test"]

    training_config = {
        "num_train_epochs": 0.064,
        "per_device_train_batch_size": 2,
    }

    trainer = tp.SequenceClassificationTrainer(
        model_name="bert-base-uncased",
        output_dir="./results",
        device="cpu",
        training_config=training_config,
    )

    # Assign the preprocessed training data to the trainer
    trainer.train_dataset = train_dataset
    trainer.val_dataset = val_dataset  # Set the validation dataset

    # Train the model
    trainer.train(from_checkpoint=False)
    trainer.save()
    metrics = trainer.get_metrics()
    print(f"Training Metrics: {metrics}")


# Function to evaluate a sequence classification model
def evaluate_sequence_classification():
    # Load and preprocess the dataset
    raw_test_dataset = load_dataset("imdb", split="test[:10]")

    tokenized_test_dataset = tp.load_data(dataset=raw_test_dataset, splits=["test"])
    test_dataset = tokenized_test_dataset["test"]

    evaluation_config = {
        "per_device_eval_batch_size": 2,
    }

    evaluator = tp.SequenceClassificationEvaluator(
        model_name="bert-base-uncased",
        device="cpu",
        evaluation_config=evaluation_config,
    )

    # Assign the preprocessed test data to the evaluator
    evaluator.data = test_dataset

    # Evaluate the model
    eval_metrics = evaluator.evaluate()
    print(f"Evaluation Metrics: {eval_metrics}")


# Function to benchmark a model
def benchmark_model():
    benchmarker = tp.Benchmarking(model_name="bert-base-uncased", device="cpu")
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
    viz = tp.Visualization()
    viz.plot_metrics(metrics)


# Function to compare models
def compare_models():
    raw_test_dataset = load_dataset("imdb", split="test[:10]")

    tokenized_test_dataset = tp.load_data(dataset=raw_test_dataset, splits=["test"])
    test_dataset = tokenized_test_dataset["test"]

    comparison = tp.ModelComparison(
        models=["bert-base-uncased", "roberta-base"],
        dataset=test_dataset,
        task="sentiment",
    )
    results = comparison.compare()
    print(f"Model Comparison Results: {results}")


# Function to explain a prediction
def explain_prediction():
    text = "I love this product!"
    explainer = tp.Explainability(
        model_name="bert-base-uncased", task="sentiment", device="cpu"
    )
    importance = explainer.feature_importance(text=text)
    print(f"Feature Importance: {importance}")


# Function to clean text using utility functions
def clean_text_example():
    raw_text = "This is some raw text! Check http://example.com"
    cleaned_text = tp.clean_text(raw_text)
    print(f"Cleaned Text: {cleaned_text}")


# Main function to run all tests
def main():
    print("Running Simple Prediction...")
    text_simple_prediction()

    # print("\nRunning Hugging Face Prediction...")
    # text_hf_prediction()

    # print("\nTraining Sequence Classification Model...")
    # train_sequence_classification()

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
