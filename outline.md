### TextPredict Syntax Documentation

This document provides a detailed guide on the syntax and structure of the `TextPredict` package, including usage examples for each method. This serves as a standard reference for contributors and users.

## Table of Contents

1. [Prediction](#prediction)
   - [Initialization](#initialization)
   - [Analyzing Text](#analyzing-text)
2. [Training](#training)
   - [Initialization](#initialization)
   - [Loading and Preprocessing Data](#loading-and-preprocessing-data)
   - [Training the Model](#training-the-model)
   - [Saving the Model](#saving-the-model)
   - [Getting Metrics](#getting-metrics)
   - [Tuning Hyperparameters](#tuning-hyperparameters)
   - [Enabling Logging](#enabling-logging)
3. [Evaluation](#evaluation)
   - [Initialization](#initialization)
   - [Loading and Preprocessing Data](#loading-and-preprocessing-data)
   - [Evaluating the Model](#evaluating-the-model)
   - [Getting Detailed Metrics](#getting-detailed-metrics)
   - [Saving Results](#saving-results)
4. [Additional Features](#additional-features)
   - [Benchmarking](#benchmarking)
   - [Visualization](#visualization)
   - [Model Comparison](#model-comparison)
   - [Configuration Management](#configuration-management)
   - [Utility Functions](#utility-functions)
   - [Custom Callbacks](#custom-callbacks)
   - [Explainability](#explainability)

## Prediction

### Initialization

```python
import textpredict as tp

# Initialize the model for sentiment analysis
model = tp.initialize(task="sentiment", device="cpu")
```

### Analyzing Text

```python
# Analyze text using the initialized model
result = model.analyze("I love this product!")
print(result)
```

## Training

### Initialization

```python
import textpredict as tp

# Initialize the trainer with model name and output directory
trainer = tp.SequenceClassificationTrainer(model_name="bert-base-uncased", output_dir="./results", device="cpu")
```

### Loading and Preprocessing Data

```python
# Load and preprocess data
data = tp.load_data(dataset_name="imdb", split="train", tokenizer="bert-base-uncased")
trainer.data = data
trainer.preprocess_data(tokenizer="bert-base-uncased")
```

### Training the Model

```python
# Train the model with or without a checkpoint
trainer.train(from_checkpoint=False)
```

### Saving the Model

```python
# Save the trained model to the output directory
trainer.save()
```

### Getting Metrics

```python
# Retrieve training metrics
metrics = trainer.get_metrics()
print(metrics)
```

### Tuning Hyperparameters

```python
# Tune hyperparameters for the model
best_params = trainer.tune_hyperparameters(train_data, eval_data, n_trials=50)
print(best_params)
```

### Enabling Logging

```python
# Enable logging for experiment tracking
trainer.enable_logging(tool="wandb", project_name="textpredict_experiments")
```

## Evaluation

### Initialization

```python
import textpredict as tp

# Initialize the evaluator with model name
evaluator = tp.SequenceClassificationEvaluator(model_name="bert-base-uncased", device="cpu")
```

### Loading and Preprocessing Data

```python
# Load and preprocess data
data = tp.load_data(dataset_name="imdb", split="test", tokenizer="bert-base-uncased")
evaluator.data = data
evaluator.preprocess_data(tokenizer="bert-base-uncased")
```

### Evaluating the Model

```python
# Evaluate the model on the preprocessed data
eval_metrics = evaluator.evaluate(data=evaluator.data)
print(eval_metrics)
```

### Getting Detailed Metrics

```python
# Retrieve detailed evaluation metrics
detailed_metrics = evaluator.get_detailed_metrics()
print(detailed_metrics)
```

### Saving Results

```python
# Save evaluation results to a file
evaluator.save_results(file_path="results.json")
```

## Additional Features

### Benchmarking

```python
import textpredict as tp

# Initialize the benchmarker with model name and device
benchmarker = tp.Benchmarking(model_name="bert-base-uncased", device="cpu")

# Benchmark the model performance
performance = benchmarker.benchmark(dataset=evaluator.data)
print(performance)
```

### Visualization

```python
import textpredict as tp

# Initialize the visualization tool
viz = tp.Visualization()

# Plot metrics
viz.plot_metrics(metrics)

# Show confusion matrix
viz.show_confusion_matrix(confusion_matrix, labels=["Positive", "Negative"])
```

### Model Comparison

```python
import textpredict as tp

# Initialize the model comparison tool
comparison = tp.ModelComparison(models=["bert-base-uncased", "roberta-base"], dataset=evaluator.data, task="sentiment")

# Compare models and print results
results = comparison.compare()
print(results)
```

### Configuration Management

```python
import textpredict as tp

# Save model configuration to a file
tp.save_config(config_path="config.json")

# Load model configuration from a file
tp.load_config(config_path="config.json")
```

### Utility Functions

```python
import textpredict as tp

# Clean text
cleaned_text = tp.clean_text("This is some raw text.")
print(cleaned_text)

# Tokenize text using a specified tokenizer
tokens = tp.tokenize_text(cleaned_text, tokenizer="bert-base-uncased")
print(tokens)
```

### Custom Callbacks

```python
import textpredict as tp

# Define a custom callback function
def custom_callback():
    print("Custom callback executed")

# Add the custom callback to the trainer
tp.add_callback(trainer, custom_callback)
```

### Explainability

```python
import textpredict as tp

# Initialize the explainability tool with model name and device
explainer = tp.Explainability(model_name="bert-base-uncased", device="cpu")

# Get feature importance for a given text
importance = explainer.feature_importance(text="I love this product!")
print(importance)
```

## Package Structure

```plaintext
textpredict/
├── __init__.py
├── benchmarking.py
├── config.py
├── datasets.py
├── evaluator.py
├── logger.py
├── model_loader.py
├── predictor.py
├── trainer/
│   ├── __init__.py
│   ├── base_trainer.py
│   ├── sequence_classification_trainer.py
│   ├── seq2seq_trainer.py
│   ├── token_classification_trainer.py
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── error_handling.py
│   ├── evaluation.py
│   ├── fine_tuning.py
│   ├── hyperparameter_tuning.py
│   ├── visualization.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── emotion.py
│   ├── sentiment.py
│   ├── zeroshot.py
```

## Contributing

To contribute to the TextPredict project, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

TextPredict is licensed under the MIT License. See the LICENSE file for more information.

