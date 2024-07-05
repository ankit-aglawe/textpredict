# textpredict/config.py

model_config = {
    "sentiment": {
        "default": "distilbert-base-uncased-finetuned-sst-2-english",
        "options": [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "nlptown/bert-base-multilingual-uncased-sentiment",
        ],
    },
    "emotion": {
        "default": "bhadresh-savani/bert-base-uncased-emotion",
        "options": [
            "bhadresh-savani/bert-base-uncased-emotion",
            "j-hartmann/emotion-english-distilroberta-base",
        ],
    },
    "zeroshot": {
        "default": "facebook/bart-large-mnli",
        "options": ["facebook/bart-large-mnli", "valhalla/distilbart-mnli-12-3"],
    },
}
