import logging
from functools import wraps

from textpredict.config import model_config, supported_tasks
from textpredict.logger import get_logger
from textpredict.model_loader import load_model, load_model_from_directory
from textpredict.utils.data_preprocessing import clean_text
from textpredict.utils.error_handling import ModelError, log_and_raise

logger = get_logger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)


def validate_task(func):
    @wraps(func)
    def wrapper(self, text, *args, **kwargs):
        if self.current_task not in self.supported_tasks:
            message = f"Unsupported task '{self.current_task}'. Supported tasks: {self.supported_tasks}"
            log_and_raise(ValueError, message)
        return func(self, text, *args, **kwargs)

    return wrapper


class TextPredict:
    def __init__(self, device="cpu"):
        self.supported_tasks = supported_tasks
        self.models = {}
        self.current_task = None
        self.device = device
        logger.info(
            "TextPredict initialized with supported tasks: "
            + ", ".join(self.supported_tasks)
        )

    def initialize(self, task, device="cpu", model_name=None, source="huggingface"):
        """
        Initialize the model for a specific task.

        Args:
            task (str): The task to perform (e.g., 'sentiment', 'emotion', 'zeroshot').
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
            model_name (str, optional): The model name. Defaults to None.
            source (str, optional): The source of the model ('huggingface' or 'local'). Defaults to 'huggingface'.
        """
        try:
            if task not in self.supported_tasks:
                raise ValueError(f"Unsupported task '{task}'")

            self.current_task = task
            self.device = device

            # if os.path.isdir(model_name):
            #     self.models[task] = load_model_from_directory(model_name, source)
            # else:
            self.default_model_name = model_name or model_config[task]

            self.load_model(self.default_model_name, source)

        except Exception as e:
            log_and_raise(
                ModelError, f"Error during initialization for task {task}: {e}"
            )

    def load_model(self, model_name, source="huggingface"):
        """
        Load a model for the current task.

        Args:
            model_name (str): The name of the model to load.
            source (str, optional): The source of the model ('huggingface' or 'local'). Defaults to 'huggingface'.
        """
        try:
            if model_name not in self.models:
                source_str = "local directory" if source == "local" else "HuggingFace"
                logger.info(f"Loading model '{model_name}' from {source_str}...")

                if source == "local":
                    self.models[model_name] = load_model_from_directory(
                        model_name, self.current_task
                    )
                else:
                    self.models[model_name] = load_model(model_name, self.current_task)

                self.models[model_name].model.to(self.device)
                logger.info(
                    f"Model '{model_name}' loaded successfully on {self.device}."
                )

        except Exception as e:
            log_and_raise(ModelError, f"Error loading model '{model_name}': {e}")

    @validate_task
    def analyze(self, text, return_probs=False, candidate_labels=None):
        """
        Analyze the given text.

        Args:
            text (str or list): The text to analyze.
            return_probs (bool, optional): Whether to return probabilities. Defaults to False.
            candidate_labels (list, optional): The candidate labels for zero-shot classification. Defaults to None.

        Returns:
            list: The analysis results.
        """
        try:

            model_name = self.default_model_name
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' is not loaded.")

            logger.info(f"Analyzing text for task: {self.current_task}")
            model = self.models[model_name]
            texts = [
                clean_text(t) for t in (text if isinstance(text, list) else [text])
            ]

            if self.current_task == "zeroshot":
                if candidate_labels is None:
                    raise ValueError(
                        "Candidate labels must be provided for zero-shot classification."
                    )

                predictions = model.predict(
                    texts, return_probs, candidate_labels=candidate_labels
                )
            else:
                predictions = model.predict(texts, return_probs)

            return predictions

        except Exception as e:
            log_and_raise(
                ModelError, f"Error during analysis for task {self.current_task}: {e}"
            )
