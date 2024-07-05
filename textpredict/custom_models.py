# textpredict/custom_models.py

import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class CustomModel:
    def __init__(self, model_name: str, num_labels: int):
        """
        Initialize a custom model with the specified parameters.

        Args:
            model_name (str): The name of the custom model to load.
            num_labels (int): The number of labels for classification.

        Raises:
            ValueError: If the model cannot be loaded.
        """
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Custom model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading custom model {model_name}: {e}")
            raise ValueError(f"Error loading custom model {model_name}: {e}")

    def predict(self, text: str or list):  # type: ignore
        """
        Make predictions using the custom model.

        Args:
            text (str or list): The input text or list of texts to classify.

        Returns:
            list: The prediction results.
        """
        try:
            if isinstance(text, str):
                text = [text]
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error making predictions with custom model: {e}")
            raise
