# TextPredict

TextPredict is a Python package for text classification tasks using transformer models. It supports various tasks such as sentiment analysis, emotion detection, and zero-shot classification. The package allows for easy fine-tuning and evaluation of models.

## Installation

You can install the package using Poetry. First, ensure that you have Poetry installed. Then, navigate to the package directory and install the dependencies:

```sh
pip install textpredict
```

## Usage

### Importing the Package

```python
from textpredict import TextPredict
```

### Analyzing Text

You can analyze text for different tasks such as sentiment analysis, emotion detection, and zero-shot classification.

#### Sentiment Analysis

```python
tp = TextPredict()
result = tp.analyse("I love using this package!", task="sentiment")
print(result)
```

#### Emotion Detection

```python
result = tp.analyse("I am excited about this!", task="emotion")
print(result)
```

#### Zero-Shot Classification

```python
result = tp.analyse(
    "This package is great for zero-shot learning.",
    task="zeroshot",
    class_list=["positive", "negative", "neutral"],
)
print(result)
```

### Fine-Tuning Models

You can fine-tune a model using your own dataset. Here is an example using the IMDb dataset:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Fine-tune model
tp.tune_model(
    task="sentiment",
    training_data=dataset["train"],
    eval_data=dataset["test"],
    num_train_epochs=1,
    batch_size=8,
    learning_rate=2e-5,
    early_stopping_patience=3,
)
```

### Evaluating Models

You can evaluate a model on a given evaluation dataset:

```python
metrics = tp.evaluate_model(task="sentiment", eval_data=dataset["test"])
print("Evaluation metrics:", metrics)
```

### Saving and Loading Models

You can save a fine-tuned model to a directory and load it later:

#### Saving a Model

```python
tp.save_model(task="sentiment", output_dir="./fine_tuned_sentiment_model")
```

#### Loading a Model

```python
tp.load_model(task="sentiment", model_dir="./fine_tuned_sentiment_model")
result = tp.analyse("I love using this package after fine-tuning!", task="sentiment")
print(result)
```

### Loading and Splitting Data

You can load datasets and split them for training and evaluation.

```python
from textpredict.datasets import load_data, get_dataset_splits

train_data = load_data("imdb", split="train")
test_data = load_data("imdb", split="test")
splits = get_dataset_splits("imdb")
print("Train Data Sample:", train_data[0])
print("Test Data Sample:", test_data[0])
print("Dataset Splits:", splits)
```

### Distributed Training

Set up distributed training for a model using the IMDb dataset.

```python
from textpredict.distributed_training import setup_distributed_training

dataset = load_dataset("imdb")
setup_distributed_training(
    model="bert-base-uncased",
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    output_dir="./results",
)
```

### Benchmarking

Measure inference time and memory usage of models.

```python
from textpredict.benchmarking import benchmark_model, measure_inference_time, measure_memory_usage

dataset = load_dataset("imdb", split="test[:100]")  # Use a small subset for benchmarking
model = load_model("distilbert-base-uncased", "sentiment")
inference_time = measure_inference_time(model, dataset)
memory_usage = measure_memory_usage(model, dataset)
benchmark_results = benchmark_model(model, dataset)
print("Inference Time:", inference_time)
print("Memory Usage:", memory_usage)
print("Benchmark Results:", benchmark_results)
```

### Command Line Interface (CLI)

You can also use the package from the command line.

#### Analyze Text

```sh
poetry run python -m textpredict.cli analyze "I love using this package!" --task sentiment
```

#### Options

- `--task`: The task to perform: sentiment, emotion, zeroshot, etc. (default: sentiment)
- `--model`: The model to use for the task. (default: None)
- `--class-list`: Comma-separated list of candidate labels for zero-shot classification. (default: None)
- `--log-level`: Set the logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL. (default: INFO)

### Web Interface

You can serve predictions via a web API using FastAPI.

#### Run the FastAPI App

```sh
poetry run uvicorn textpredict.web_interface:app --reload
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
