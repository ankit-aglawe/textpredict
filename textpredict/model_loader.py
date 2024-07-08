from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textpredict.logger import get_logger
from textpredict.models import EmotionModel, SentimentModel, ZeroShotModel
from textpredict.utils.error_handling import log_and_raise

logger = get_logger(__name__)

MODEL_CLASS_MAPPING = {
    "sentiment": SentimentModel,
    "emotion": EmotionModel,
    "zeroshot": ZeroShotModel,
}


logger = get_logger(__name__)


def load_model(model_name: str, task: str):
    try:
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
    try:
        logger.info(f"Loading model from directory {model_dir} for task {task}")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
