from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textpredict.task_models.base import BaseModel


class Seq2seqModel(BaseModel):
    def __init__(self, model_name: str, model=None, tokenizer=None):
        super().__init__(model_name, "seq2seq", model, tokenizer)
        if not model or not tokenizer:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, texts, return_probs=False):
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        outputs = self.model.generate(**inputs)

        decoded_outputs = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        results = [
            {"text": text, "generated_text": generated_text}
            for text, generated_text in zip(texts, decoded_outputs)
        ]

        return results

    def get_label_ids(self):
        return None
