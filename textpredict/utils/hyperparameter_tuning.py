import optuna
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from textpredict.evaluation import compute_metrics


def objective(trial, model_name, train_data, eval_data, output_dir, device):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5)
    per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 2, 8)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)
    adam_epsilon = trial.suggest_float("adam_epsilon", 1e-8, 1e-6)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train_data = train_data.map(tokenize_function, batched=True)
    tokenized_eval_data = eval_data.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        adam_epsilon=adam_epsilon,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    accuracy = eval_metrics["eval_accuracy"]

    return accuracy


def tune_hyperparameters(
    train_data, eval_data, model_name, output_dir, device, n_trials=50
):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial, model_name, train_data, eval_data, output_dir, device
        ),
        n_trials=n_trials,
    )
    best_params = study.best_params
    return best_params
