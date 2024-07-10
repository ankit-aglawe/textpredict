from transformers import AutoModelForTokenClassification, AutoTokenizer

from textpredict.task_models.base import BaseModel


class NERModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None):
        super().__init__(model_name, "ner", model, tokenizer)
        if not model or not tokenizer:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=False,
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
                if label != "O":  # "O" means the token is not part of an entity
                    entities.append({"entity": label, "token": token})

            results.append({"text": text, "entities": entities})

        return results
