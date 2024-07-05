import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textpredict.config import model_config
from textpredict.logger import get_logger
from textpredict.models import (
    EmotionModel,
    SentimentModel,
    ZeroShotModel,
)
from textpredict.utils.error_handling import log_and_raise

logger = get_logger(__name__)


def load_model(model_name: str, task: str):
    """
    Load the appropriate model for the given task.

    Args:
        model_name (str): The name of the model to load.
        task (str): The task for which the model is to be loaded.

    Returns:
        BaseModel: An instance of the appropriate model class.

    Raises:
        ValueError: If the task is not supported or the model name is not valid for the task.
    """
    try:
        if os.path.isdir(model_name):
            return load_model_from_directory(model_name, task)

        task_config = model_config.get(task)
        if not task_config:
            raise ValueError(f"Task {task} not supported.")

        if model_name not in task_config["options"]:
            raise ValueError(f"Model {model_name} not supported for task {task}.")

        logger.info(f"Loading model {model_name} for task {task}")

        model_class_mapping = {
            "sentiment": SentimentModel,
            "emotion": EmotionModel,
            "zeroshot": ZeroShotModel,
        }

        model_class = model_class_mapping.get(task)
        if not model_class:
            raise ValueError(f"No model class found for task {task}")

        return model_class(model_name)
    except Exception as e:
        log_and_raise(RuntimeError, f"Error loading model for task {task}: {e}")


def load_model_from_directory(model_dir: str, task: str):
    """
    Load a model and tokenizer from a saved directory.

    Args:
        model_dir (str): The directory from which to load the model and tokenizer.
        task (str): The task for which the model is to be loaded.

    Returns:
        BaseModel: An instance of the appropriate model class.
    """
    try:
        logger.info(f"Loading model from directory {model_dir} for task {task}")

        model = AutoModelForSequenceClassification.from_pretrained(  # noqa: F841
            model_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)  # noqa: F841

        model_class_mapping = {
            "sentiment": SentimentModel,
            "emotion": EmotionModel,
            "zeroshot": ZeroShotModel,
        }

        model_class = model_class_mapping.get(task)
        if not model_class:
            raise ValueError(f"No model class found for task {task}")

        return model_class(model_name=model_dir, model=model, tokenizer=tokenizer)
    except Exception as e:
        log_and_raise(
            RuntimeError, f"Error loading model from directory for task {task}: {e}"
        )
