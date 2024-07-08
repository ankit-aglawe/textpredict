# textpredict/models/zeroshot.py

from transformers import pipeline

from textpredict.models.base import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None):
        super().__init__(model_name, "zero-shot-classification", model, tokenizer)
        self.pipeline = pipeline(
            "zero-shot-classification", model=self.model, tokenizer=self.tokenizer
        )

    def predict(self, texts, class_list):
        results = []
        for text in texts:
            prediction = self.pipeline(text, candidate_labels=class_list)
            results.append(
                {"labels": prediction["labels"], "scores": prediction["scores"]}
            )
        return results
