<div align="center">

[![python](https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12|3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![PyPI - Version](https://img.shields.io/pypi/v/sentimentpredictor) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) [![Downloads](https://static.pepy.tech/badge/textpredict)](https://pepy.tech/project/textpredict)


![TextPredict Logo](https://raw.githubusercontent.com/ankit-aglawe/textpredict/main/assets/logo.png)

## Advanced Text Classification with Transformer Models
</div>

**TextPredict** is a powerful Python package designed for text classification tasks leveraging advanced transformer models. It supports a variety of tasks including sentiment analysis, emotion detection, and zero-shot classification, making it an essential tool for developers and data scientists working in natural language processing (NLP), machine learning (ML), and artificial intelligence (AI).

### Features

- **Sentiment Analysis**: Classify text based on sentiment with high accuracy.
- **Emotion Detection**: Detect emotions in text effortlessly.
- **Zero-Shot Classification**: Classify text into unseen categories without any additional training.
- **Fine-Tuning**: Easily fine-tune models to improve performance on specific datasets.
- **Model Evaluation**: Evaluate model performance with robust metrics.
- **Distributed Training**: Support for distributed training to leverage multiple GPUs.

### Installation

```sh
pip install textpredict
```

### Usage

#### Sentiment Analysis

```python
from textpredict import TextPredict

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

#### Fine-Tuning Models

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

#### Evaluating Models

```python
metrics = tp.evaluate_model(task="sentiment", eval_data=dataset["test"])
print("Evaluation metrics:", metrics)
```

#### Saving and Loading Models

```python
tp.save_model(task="sentiment", output_dir="./fine_tuned_sentiment_model")
tp.load_model(task="sentiment", model_dir="./fine_tuned_sentiment_model")
result = tp.analyse("I love using this package after fine-tuning!", task="sentiment")
print(result)
```


### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on GitHub.

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Links

- **GitHub Repository**: [Github](https://github.com/ankit-aglawe/textpredict)
- **PyPI Project**: [PYPI](https://pypi.org/project/textpredict/)
- **Documentation**: [Readthedocs](https://github.com/ankit-aglawe/sentimentpredictor#readme)
- **Source Code**: [Source Code](https://github.com/ankit-aglawe/sentimentpredictor)
- **Issue Tracker**: [Issue Tracker](https://github.com/ankit-aglawe/sentimentpredictor/issues)
