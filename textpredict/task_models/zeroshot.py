from transformers import pipeline

from textpredict.task_models.base import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None):
        super().__init__(model_name, "zero-shot-classification", model, tokenizer)
        self.pipeline = pipeline(
            "zero-shot-classification", model=self.model, tokenizer=self.tokenizer
        )

    def predict(self, texts, candidate_labels):
        if not isinstance(candidate_labels, list) or not all(
            isinstance(label, str) for label in candidate_labels
        ):
            raise ValueError("candidate_labels must be a list of strings")

        results = []
        for text in texts:
            prediction = self.pipeline(text, candidate_labels=candidate_labels)
            results.append(
                {"labels": prediction["labels"], "scores": prediction["scores"]}
            )
        return results
