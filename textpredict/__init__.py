from textpredict.logger import get_logger

from .benchmarking import Benchmarking
from .config import load_config, save_config
from .datasets import load_data
from .device_manager import DeviceManager
from .evaluators import (
    Seq2seqEvaluator,
    SequenceClassificationEvaluator,
    TokenClassificationEvaluator,
)
from .explainability import Explainability
from .model_comparison import ModelComparison
from .predictor import TextPredict
from .trainers import (
    Seq2seqTrainer,
    SequenceClassificationTrainer,
    TokenClassificationTrainer,
)
from .utils import clean_text, tokenize_text
from .visualization import Visualization

__all__ = [
    "TextPredict",
    "SequenceClassificationTrainer",
    "Seq2seqTrainer",
    "TokenClassificationTrainer",
    "SequenceClassificationEvaluator",
    "Seq2seqEvaluator",
    "TokenClassificationEvaluator",
    "Benchmarking",
    "Visualization",
    "ModelComparison",
    "Explainability",
    "save_config",
    "load_config",
    "load_data",
    "clean_text",
    "tokenize_text",
]


def initialize(task, model_name=None, source="huggingface", device=None):
    predictor = TextPredict(device=device)
    predictor.initialize(task=task, model_name=model_name, source=source)
    return predictor


def set_device(device):
    DeviceManager.set_device(device)
