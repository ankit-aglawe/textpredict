import json

model_config = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "emotion": "SamLowe/roberta-base-go_emotions",
    "zeroshot": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "ner": "dslim/bert-base-NER",
}

default_training_config = {
    "learning_rate": 3e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 100,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,
    "warmup_steps": 500,
    "adam_epsilon": 1e-8,
}

default_evaluation_config = {
    "per_device_eval_batch_size": 8,
    "output_dir": "./results",
}


def save_config(config, file_path):
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)


def load_config(file_path):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config
