# Usage

## Installation

You can install the package via pip:

```sh
pip install textpredict
```

## Quick Start

### Initialization and Simple Prediction

```python
import textpredict as tp

# Initialize for sentiment analysis
model = tp.initialize(task="sentiment")
texts = ["I love this product!", "I hate this product!"]
result = model.analyze(texts, return_probs=False)
print(f"Sentiment Prediction Result: {result}")
```

### Using Pre-trained Models from Hugging Face

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

### Using Models from Local Directory

```python
model = tp.initialize(
    task="sentiment",
    model_name="./results",
    source="local",
)
text = "I love this product!"
result = model.analyze(text, return_probs=True)
print(f"Sentiment Prediction Result: {result}")
```

### Training a Model

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