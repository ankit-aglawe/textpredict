# API Reference

## `textpredict.initialize`

Initialize the TextPredict model for a specific task.

### Parameters

- `task` (str): The task to perform (e.g., 'sentiment', 'emotion', 'zeroshot', 'ner', 'sequence_classification', 'token_classification', 'seq2seq').
- `device` (str, optional): The device to run the model on. Defaults to 'cpu'.
- `model_name` (str, optional): The model name. Defaults to None.
- `source` (str, optional): The source of the model ('huggingface' or 'local'). Defaults to 'huggingface'.

### Returns

An initialized TextPredict model.

## `textpredict.model.analyze`

Analyze the provided texts using the initialized model.

### Parameters

- `texts` (list of str): The texts to analyze.
- `return_probs` (bool, optional): Whether to return probabilities along with predictions. Defaults to False.

### Returns

Analysis results for the provided texts.

## `textpredict.SequenceClassificationTrainer`

Trainer class for sequence classification models.

### Parameters

- `model_name` (str): The name of the model to use.
- `output_dir` (str): The directory to save the trained model.
- `train_dataset` (Dataset): The dataset to use for training.
- `val_dataset` (Dataset): The dataset to use for validation.

### Methods

- `train(from_checkpoint=True)`: Train the model.
- `save()`: Save the trained model.
- `evaluate(test_dataset)`: Evaluate the model on the test dataset.

## `textpredict.Explainability`

Class for model explainability and feature importance.

### Parameters

- `model_name` (str): The name of the model to use.
- `task` (str): The task to perform.
- `device` (str, optional): The device to run the model on. Defaults to 'cpu'.

### Methods

- `feature_importance(text)`: Get feature importance for the given text.