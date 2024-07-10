# Examples

## Using Pre-trained Models from Hugging Face

```python
model = tp.initialize(
    task="sentiment",
    device="cpu",
    model_name="AnkitAI/reviews-roberta-base-sentiment-analysis",
    source="huggingface",
)
text = "I love this product!"
result = model.analyze(text, return_probs=True)
print(f"Sentiment Prediction Result: {result}")
```

## Training a Sequence Classification Model

```python
import textpredict as tp
from datasets import load_dataset

# Load dataset
train_data = load_dataset("imdb", split="train[:10]")
val_data = load_dataset("imdb", split="test[:10]")

# Initialize and train the model
trainer = tp.SequenceClassificationTrainer(
    model_name="bert-base-uncased",
    output_dir="./results",
    train_dataset=train_data,
    val_dataset=val_data
)
trainer.train()

# Save the trained model
trainer.save()

# Evaluate the model
metrics = trainer.evaluate(test_dataset=val_data)
print(f"Evaluation Metrics: {metrics}")
```

## Explainability and Feature Importance

```python
text = "I love this product!"
explainer = tp.Explainability(model_name="bert-base-uncased", task="sentiment", device="cpu")
importance = explainer.feature_importance(text=text)
print(f"Feature Importance: {importance}")
```

## Benchmarking a Model

```python
from datasets import load_dataset
import textpredict as tp

def benchmark_model():
    benchmarker = tp.Benchmarking(model_name="bert-base-uncased", device="cpu")
    dataset = load_dataset("imdb", split="test[:10]")
    dataset = dataset.map(lambda x: benchmarker.tokenizer(x["text"], padding="max_length", truncation=True), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    performance = benchmarker.benchmark(dataset)
    print(f"Benchmarking Performance: {performance}")

benchmark_model()
```

## Visualizing Metrics

```python
import textpredict as tp

def visualize_metrics():
    metrics = {"accuracy": [0.8, 0.85, 0.9], "loss": [0.6, 0.4, 0.2]}
    viz = tp.Visualization()
    viz.plot_metrics(metrics)

visualize_metrics()
```

## Comparing Models

```python
from datasets import load_dataset
import textpredict as tp

def compare_models():
    raw_test_dataset = load_dataset("imdb", split="test[:10]")
    tokenized_test_dataset = tp.load_data(dataset=raw_test_dataset, splits=["test"])
    test_dataset = tokenized_test_dataset["test"]

    comparison = tp.ModelComparison(models=["bert-base-uncased", "roberta-base"], dataset=test_dataset, task="sentiment")
    results = comparison.compare()
    print(f"Model Comparison Results: {results}")

compare_models()
```
