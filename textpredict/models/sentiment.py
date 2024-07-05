# textpredict/models/sentiment.py

from textpredict.models.base import BaseModel


class SentimentModel(BaseModel):
    def __init__(self, model_name: str):
        """
        Initialize the SentimentModel with the specified parameters.

        Args:
            model_name (str): The name of the model to load.
        """
        super().__init__(model_name, "sentiment-analysis")
