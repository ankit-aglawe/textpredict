# textpredict/models/zeroshot.py

from textpredict.models.base import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, model_name: str):
        """
        Initialize the ZeroShotModel with the specified parameters.

        Args:
            model_name (str): The name of the model to load.
        """
        super().__init__(model_name, "zero-shot-classification", multi_label=True)

    def predict(self, text: str or list, class_list: list = None) -> list:  # type: ignore
        """
        Make predictions using the zero-shot classification model.

        Args:
            text (str or list): The input text or list of texts to classify.
            class_list (list, optional): The list of candidate labels for zero-shot classification.

        Returns:
            list: The prediction results.

        Raises:
            ValueError: If class_list is not provided.
        """
        if not class_list:
            raise ValueError(
                "Class list must be provided for zero-shot classification."
            )
        return super().predict(text, class_list=class_list)
