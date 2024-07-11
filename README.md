<div align="center">

[![python](https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12|3.13-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![PyPI - Version](https://img.shields.io/pypi/v/sentimentpredictor) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) [![Downloads](https://static.pepy.tech/badge/textpredict)](https://pepy.tech/project/textpredict)


![TextPredict Logo](https://raw.githubusercontent.com/ankit-aglawe/textpredict/main/assets/logo3.png)

## Advanced Text Classification with Transformer Models
</div>
TextPredict is a powerful Python package designed for various text analysis and prediction tasks using advanced NLP models. It simplifies the process of performing sentiment analysis, emotion detection, zero-shot classification, named entity recognition (NER), and more. Built on top of Hugging Face's Transformers, TextPredict allows seamless integration with pre-trained models or custom models for specific tasks.

## Features

- **Sentiment Analysis**: Determine the sentiment of text (positive, negative, neutral).
- **Emotion Detection**: Identify emotions such as happiness, sadness, anger, etc.
- **Zero-Shot Classification**: Classify text into custom categories without additional training.
- **Named Entity Recognition (NER)**: Extract entities like names, locations, and organizations from text.
- **Sequence Classification**: Fine-tune models for custom classification tasks.
- **Token Classification**: Classify tokens within text for tasks like NER.
- **Sequence-to-Sequence (Seq2Seq)**: Perform tasks like translation and summarization.
- **Model Comparison**: Evaluate and compare multiple models on the same dataset.
- **Explainability**: Understand model predictions through feature importance analysis.
- **Text Cleaning**: Utilize utility functions for preprocessing text data.

## Supported Tasks

- Sentiment Analysis
- Emotion Detection
- Zero-Shot Classification
- Named Entity Recognition (NER)
- Sequence Classification
- Token Classification
- Sequence-to-Sequence (Seq2Seq)

## Installation

You can install the package via pip:

```sh
pip install textpredict
```

## Quick Start

### Initialization and Simple Prediction

Initialize the TextPredict model and perform simple predictions:

```python
import textpredict as tp

# Initialize for sentiment analysis

# task : ["sentiment", "ner", "zeroshot", "emotion", "sequence_classification", "token_classification", "seq2seq" etc]

model = tp.initialize(task="sentiment") 
result = model.analyze(text = ["I love this product!", "I hate this product!"], return_probs=False)
```

### Using Pre-trained Models from Hugging Face

Utilize a specific pre-trained model from Hugging Face:

```python
model = tp.initialize(task="emotion", model_name="AnkitAI/reviews-roberta-base-sentiment-analysis", source="huggingface")
result = model.analyze(text = "I love this product!", return_probs=True)
```

### Using Models from Local Directory

Load and use a model from a local directory:

```python
model = tp.initialize(task="ner", model_name="./results", source="local")
result = model.analyze(text="I love this product!", return_probs=True)
```

### Training a Model

Train a model for sequence classification:

```python
import textpredict as tp
from datasets import load_dataset

# Load dataset
train_data = load_dataset("imdb", split="train")
val_data = load_dataset("imdb", split="test")

# Initialize and train the model
trainer = tp.SequenceClassificationTrainer(model_name="bert-base-uncased", output_dir="./results", train_dataset=train_data, val_dataset=val_data)
trainer.train()

# Save and evaluate the trained model
trainer.save()
metrics = trainer.evaluate(test_dataset=val_data)
```

For detailed examples, refer to the `examples` directory.

### Explainability and Feature Importance

Understand model predictions with feature importance:

```python
text = "I love this product!"
explainer = tp.Explainability(model_name="bert-base-uncased", task="sentiment", device="cpu")
importance = explainer.feature_importance(text=text)
```

## Documentation

For detailed documentation, please refer to the [TextPredict Documentation](https://ankit-aglawe.github.io/textpredict/).

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before making a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

This project leverages the [Transformers](https://github.com/huggingface/transformers) library by Hugging Face. We extend our gratitude to the Hugging Face team and to the developers, contributors for their work for their work in creating and maintaining such a valuable resource for the NLP community.


### Links

- **GitHub Repository**: [Github](https://github.com/ankit-aglawe/textpredict)
- **PyPI Project**: [PYPI](https://pypi.org/project/textpredict/)
- **Documentation**: [TextPredict Documentation](https://ankit-aglawe.github.io/textpredict/)
- **Source Code**: [Source Code](https://github.com/ankit-aglawe/sentimentpredictor)
- **Issue Tracker**: [Issue Tracker](https://github.com/ankit-aglawe/sentimentpredictor/issues)


