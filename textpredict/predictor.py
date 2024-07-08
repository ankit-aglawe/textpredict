import logging
from functools import wraps

from textpredict.config import model_config
from textpredict.logger import get_logger
from textpredict.model_loader import load_model, load_model_from_directory
from textpredict.utils.data_preprocessing import clean_text
from textpredict.utils.error_handling import ModelError, log_and_raise

logger = get_logger(__name__)

# Suppress verbose logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


def validate_task(func):
    @wraps(func)
    def wrapper(self, text, task, *args, **kwargs):
        if task not in self.supported_tasks:
            message = (
                f"Unsupported task '{task}'. Supported tasks: {self.supported_tasks}"
            )
            log_and_raise(ValueError, message)
        return func(self, text, task, *args, **kwargs)

    return wrapper


class TextPredict:
    def __init__(self, model_name=None, device="cpu"):
        self.supported_tasks = list(model_config.keys())
        self.models = {}
        self.default_model_name = model_name
        self.device = device
        logger.info(
            "TextPredict initialized with supported tasks: "
            + ", ".join(self.supported_tasks)
        )

    def load_model_if_not_loaded(self, task, model_name=None, from_local=True):
        if task not in self.models:
            model_name = model_name or self.default_model_name or model_config[task]
            source = "local directory" if from_local else "HuggingFace"
            logger.info(
                f"Loading model '{model_name}' for task '{task}' from {source}..."
            )
            if from_local:
                self.models[task] = load_model_from_directory(model_name, task)
            else:
                self.models[task] = load_model(model_name, task)
            self.models[task].model.to(self.device)
            logger.info(
                f"Model for task '{task}' loaded successfully on {self.device}."
            )

    @validate_task
    def analyse(self, text, task, class_list=None, return_probs=False):
        try:
            self.load_model_if_not_loaded(task)
            logger.info(f"Analyzing text for task: {task}")
            model = self.models[task]

            texts = [
                clean_text(t) for t in (text if isinstance(text, list) else [text])
            ]

            if task == "zeroshot":
                predictions = model.predict(texts, class_list)
            else:
                predictions = model.predict(texts)

            if return_probs:
                return predictions

            return [
                pred["label"] if "label" in pred else pred["labels"]
                for pred in predictions
            ]
        except Exception as e:
            log_and_raise(ModelError, f"Error during analysis for task {task}: {e}")

    def load_model(self, task, model_dir, from_local=True):
        try:
            logger.info(f"Loading model for task: {task} from {model_dir}")
            if from_local:
                self.models[task] = load_model_from_directory(model_dir, task)
            else:
                self.models[task] = load_model(model_dir, task)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            log_and_raise(ModelError, f"Error loading model for task {task}: {e}")
