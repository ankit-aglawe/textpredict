# textpredict/models/emotion.py

from textpredict.task_models.base import BaseModel


class EmotionModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None):
        super().__init__(model_name, "emotion", model, tokenizer)

    def predict(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).tolist()
        labels = [self.model.config.id2label[p] for p in predictions]
        return [{"label": label} for label in labels]
