from transformers import AutoModelForTokenClassification, AutoTokenizer

from textpredict.logger import get_logger
from textpredict.task_models.base import BaseModel

logger = get_logger(__name__)


class NERModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None, device=None):
        super().__init__(model_name, "ner", model, tokenizer, device)

        self.device = device

        if not model or not tokenizer:
            try:
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.model.to(self.device)
            except Exception as e:
                logger.warnings(f"Failed to load model on {self.device}: {e}")
                logger.info("Falling back to CPU.")
                self.device = "cpu"
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, texts, return_probs=False):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=False,  # Set to False for full tokenization
        )
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        results = []
        for i, text in enumerate(texts):
            entities = []
            input_ids = inputs.input_ids[i]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            labels = [self.model.config.id2label[p] for p in predictions[i].tolist()]

            for token, label in zip(tokens, labels):
                if label != "O":
                    entities.append({"entity": label, "token": token})

            results.append({"text": text, "entities": entities})

        return results
