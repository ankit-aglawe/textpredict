# textpredict/models/sentiment.py

from textpredict.task_models.base import BaseModel


class SentimentModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None, device=None):
        super().__init__(model_name, "sentiment", model, tokenizer, device)

    def predict(self, texts, return_probs):
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        outputs = self.model(**inputs)

        predictions = outputs.logits.argmax(dim=-1).tolist()

        labels = [self.model.config.id2label[p] for p in predictions]

        if return_probs:
            logits = outputs.logits
            probabilities = logits.softmax(dim=-1)

            return [
                {"label": label, "score": max(prob), "probabilities": prob}
                for label, prob in zip(labels, probabilities.tolist())
            ]
        else:
            return [{"label": label} for label in labels]

    def get_label_ids(self):
        label_ids = list(self.model.config.id2label.keys())
        labels = list(self.model.config.id2label.values())
        return [
            {"label_id": label_id, "label": label}
            for label_id, label in zip(label_ids, labels)
        ]
