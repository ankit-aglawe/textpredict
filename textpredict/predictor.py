import logging
from functools import wraps

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from textpredict.config import model_config
from textpredict.logger import get_logger
from textpredict.model_loader import load_model, load_model_from_directory
from textpredict.utils.data_preprocessing import clean_text
from textpredict.utils.error_handling import ModelError, log_and_raise
from textpredict.utils.evaluation import compute_metrics, log_metrics
from textpredict.utils.fine_tuning import fine_tune_model, tokenize_and_encode

logger = get_logger(__name__)

# Suppress verbose logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


def validate_task(func):
    @wraps(func)
    def wrapper(self, text, task, *args, **kwargs):
        if task not in self.supported_tasks:
            message = (
                f"Unsupported task '{task}'. Supported tasks: {self.supported_tasks}"
            )
            log_and_raise(ValueError, message)
        return func(self, text, task, *args, **kwargs)

    return wrapper


class TextPredict:
    def __init__(self, model_name=None):
        """
        Initialize the TextPredict class with supported tasks but do not load models.

        Args:
            model_name (str, optional): The name of the model to load. If not provided, default models will be used.
        """
        try:
            self.supported_tasks = list(model_config.keys())
            self.models = {}
            self.default_model_name = model_name
            logger.info(
                "TextPredict initialized with supported tasks: "
                + ", ".join(self.supported_tasks)
            )
        except Exception as e:
            log_and_raise(ModelError, f"Error initializing TextPredict: {e}")

    def load_model_if_not_loaded(self, task):
        if task not in self.models:
            config = model_config.get(task)
            if config:
                model_name = self.default_model_name or config["default"]
                logger.info(f"Loading model for task '{task}'...")
                self.models[task] = load_model(model_name, task)
                logger.info(f"Model for task '{task}' loaded successfully.")
            else:
                log_and_raise(ValueError, f"No configuration found for task '{task}'.")

    @validate_task
    def analyse(self, text, task, class_list=None, return_probs=False):
        """
        Analyze the given text for the specified task.

        Args:
            text (str or list): The input text or list of texts to analyze.
            task (str): The task for which to analyze the text.
            class_list (list, optional): The list of candidate labels for zero-shot classification.
            return_probs (bool, optional): Whether to return prediction probabilities. Defaults to False.

        Returns:
            list: The analysis results.
        """
        try:
            self.load_model_if_not_loaded(task)
            logger.info(f"Analyzing text for task: {task}")
            model = self.models[task]
            if isinstance(text, str):
                text = [clean_text(text)]
            elif isinstance(text, list):
                text = [clean_text(t) for t in text]
            else:
                log_and_raise(
                    TypeError, "Input text must be a string or a list of strings."
                )
            if task == "zeroshot":
                predictions = model.predict(text, class_list)
            else:
                predictions = model.predict(text)
            if return_probs:
                return predictions
            try:
                return [pred["label"] for pred in predictions]
            except:
                return [pred["labels"] for pred in predictions]
        except Exception as e:
            log_and_raise(ModelError, f"Error during analysis for task {task}: {e}")

    def tune_model(
        self, task, training_data, eval_data=None, output_dir="./results", **kwargs
    ):
        """
        Fine-tune the model for the specified task.

        Args:
            task (str): The task for which to fine-tune the model.
            training_data (Dataset): The training data to use for fine-tuning.
            eval_data (Dataset, optional): The evaluation data to use for validation. Defaults to None.
            output_dir (str, optional): The directory to save the fine-tuned model and checkpoints. Defaults to "./results".
            kwargs: Additional keyword arguments for TrainingArguments.
        """
        try:
            self.load_model_if_not_loaded(task)
            logger.info(f"Fine-tuning model for task: {task}")
            model_pipeline = self.models[task]
            model = model_pipeline.model
            tokenizer = model_pipeline.tokenizer
            fine_tune_model(
                model, tokenizer, training_data, eval_data, output_dir, **kwargs
            )
        except Exception as e:
            log_and_raise(ModelError, f"Error during fine-tuning for task {task}: {e}")

    def evaluate_model(self, task, eval_data, **kwargs):
        """
        Evaluate the model on the provided evaluation data.

        Args:
            task (str): The task for which to evaluate the model.
            eval_data (Dataset): The dataset to use for evaluation.
            kwargs: Additional keyword arguments for TrainingArguments.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        try:
            self.load_model_if_not_loaded(task)
            logger.info(f"Evaluating model for task: {task}")
            model_pipeline = self.models[task]
            model = model_pipeline.model
            tokenizer = model_pipeline.tokenizer

            # Tokenize and encode the evaluation dataset
            eval_data = tokenize_and_encode(eval_data, tokenizer)

            data_collator = DataCollatorWithPadding(tokenizer)
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_eval_batch_size=kwargs.get("eval_batch_size", 8),
                logging_dir="./logs",
                logging_steps=kwargs.get("logging_steps", 200),
                eval_steps=kwargs.get("eval_steps", 200),
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=eval_data,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            eval_metrics = trainer.evaluate()
            log_metrics(eval_metrics)
            return eval_metrics
        except Exception as e:
            log_and_raise(ModelError, f"Error during evaluation for task {task}: {e}")

    def save_model(self, task, output_dir):
        """
        Save the model for the specified task to the given directory.

        Args:
            task (str): The task for which to save the model.
            output_dir (str): The directory to save the model.
        """
        try:
            self.load_model_if_not_loaded(task)
            logger.info(f"Saving model for task: {task} to {output_dir}")
            model_pipeline = self.models[task]
            model_pipeline.model.save_pretrained(output_dir)
            model_pipeline.tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        except Exception as e:
            log_and_raise(ModelError, f"Error saving model for task {task}: {e}")

    def load_model(self, task, model_dir):
        """
        Load a model for the specified task from the given directory.

        Args:
            task (str): The task for which to load the model.
            model_dir (str): The directory to load the model from.
        """
        try:
            logger.info(f"Loading model for task: {task} from {model_dir}")
            self.models[task] = load_model_from_directory(model_dir, task)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            log_and_raise(ModelError, f"Error loading model for task {task}: {e}")


# # Example usage
# if __name__ == "__main__":
#     from datasets import load_dataset

#     tp = TextPredict()
#     print(tp.analyse("I love using this package!", task="sentiment"))
#     print(tp.analyse("I am excited about this!", task="emotion"))
#     print(
#         tp.analyse(
#             "This package is great for zero-shot learning.",
#             task="zeroshot",
#             class_list=["positive", "negative", "neutral"],
#         )
#     )

#     # Fine-tuning example (assuming we have a dataset)
#     dataset = load_dataset("imdb")
#     tp.tune_model(
#         task="sentiment",
#         training_data=dataset["train"],
#         eval_data=dataset["test"],
#         num_train_epochs=1,
#         batch_size=8,
#         learning_rate=2e-5,
#         early_stopping_patience=3,
#     )

#     # Evaluate the fine-tuned model
#     metrics = tp.evaluate_model(task="sentiment", eval_data=dataset["test"])
#     print("Evaluation metrics:", metrics)

#     # Save the fine-tuned model
#     tp.save_model(task="sentiment", output_dir="./fine_tuned_sentiment_model")

#     # Load the saved model
#     tp.load_model(task="sentiment", model_dir="./fine_tuned_sentiment_model")
#     print(tp.analyse("I love using this package after fine-tuning!", task="sentiment"))
