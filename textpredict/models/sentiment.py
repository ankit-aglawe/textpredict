# textpredict/models/sentiment.py

from textpredict.models.base import BaseModel


class SentimentModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None):
        super().__init__(model_name, "sentiment-analysis", model, tokenizer)

    def predict(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).tolist()
        labels = [self.model.config.id2label[p] for p in predictions]
        return [{"label": label} for label in labels]
