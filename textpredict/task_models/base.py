from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textpredict.logger import get_logger

logger = get_logger(__name__)


class BaseModel:
    def __init__(self, model_name, task_type, model=None, tokenizer=None, device=None):
        self.model_name = model_name
        self.task_type = task_type
        self.device = device

        if model and tokenizer:  # for sequencial classification tasks
            self.model = model
            self.tokenizer = tokenizer
        else:
            try:
                # for task "sentiment", "emotion" etc
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self.model.to(self.device)

            except Exception as e:
                logger.warnings(f"Failed to load model on {self.device}: {e}")
                logger.info("Falling back to CPU.")
                self.device = "cpu"
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, texts, return_probs, candidate_labels=None):
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
