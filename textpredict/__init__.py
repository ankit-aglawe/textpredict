# textpredict/__init__.py

from .config import model_config
from .predictor import TextPredict

__all__ = ["TextPredict", "model_config"]
