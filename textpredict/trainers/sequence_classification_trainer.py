from transformers import AutoModelForSequenceClassification

from .base_trainer import BaseTrainer


class SequenceClassificationTrainer(BaseTrainer):
    def load_model(self, model_name):
        return AutoModelForSequenceClassification.from_pretrained(model_name)
