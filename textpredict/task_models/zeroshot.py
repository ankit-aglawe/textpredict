from transformers import pipeline

from textpredict.task_models.base import BaseModel


class ZeroShotModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None, device=None):
        super().__init__(
            model_name, "zero-shot-classification", model, tokenizer, device
        )
        self.pipeline = pipeline(
            "zero-shot-classification", model=self.model, tokenizer=self.tokenizer
        )

    def predict(self, texts, return_probs=False, candidate_labels=None):
        if not isinstance(candidate_labels, list) or not all(
            isinstance(label, str) for label in candidate_labels
        ):
            raise ValueError("candidate_labels must be a list of strings")

        predictions = self.pipeline(
            texts, candidate_labels=candidate_labels, multi_label=True
        )

        if return_probs:
            return [
                {
                    "label": prediction["labels"][
                        prediction["scores"].index(max(prediction["scores"]))
                    ],
                    "score": max(prediction["scores"]),
                    "probabilities": prediction["scores"],
                }
                for prediction in predictions
            ]
        else:
            return [
                {
                    "label": prediction["labels"][
                        prediction["scores"].index(max(prediction["scores"]))
                    ],
                }
                for prediction in predictions
            ]
