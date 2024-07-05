import logging

from textpredict.utils.error_handling import log_and_raise

logger = logging.getLogger(__name__)


def predict(model, text, task, class_list=None):
    """
    Predict the result based on the task and model.

    Args:
        model: The model to use for prediction.
        text (str or list): The input text(s) to predict.
        task (str): The task type (e.g., 'sentiment', 'emotion', 'zeroshot').
        class_list (list, optional): The list of candidate labels for zero-shot classification.

    Returns:
        The prediction result.

    Raises:
        ValueError: If the task is not supported or class_list is not provided for zero-shot.
    """
    try:
        if task == "sentiment":
            return model(text)
        elif task == "emotion":
            return model(text)
        elif task == "zeroshot":
            if class_list is None:
                raise ValueError(
                    "Class list must be provided for zero-shot classification."
                )
            return model(text, candidate_labels=class_list)
        else:
            log_and_raise(ValueError, f"Task {task} not supported.")
    except Exception as e:
        log_and_raise(RuntimeError, f"Error during prediction: {e}")
