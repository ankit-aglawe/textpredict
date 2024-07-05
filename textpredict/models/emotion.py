# textpredict/models/emotion.py

from textpredict.models.base import BaseModel


class EmotionModel(BaseModel):
    def __init__(self, model_name: str):
        """
        Initialize the EmotionModel with the specified parameters.

        Args:
            model_name (str): The name of the model to load.
        """
        super().__init__(model_name, "text-classification")
