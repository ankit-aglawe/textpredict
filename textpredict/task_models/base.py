from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BaseModel:
    def __init__(self, model_name, task_type, model=None, tokenizer=None):
        self.model_name = model_name
        self.task_type = task_type
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, texts):
        raise NotImplementedError(
            "The predict method must be implemented by subclasses."
        )

    def format_output(self, predictions):
        formatted_results = []
        for prediction in predictions:
            formatted_results.append(
                {
                    "text": prediction.get("text"),
                    "labels": prediction.get("labels", []),
                    "scores": prediction.get("scores", []),
                    "entities": prediction.get("entities", []),
                }
            )
        return formatted_results
