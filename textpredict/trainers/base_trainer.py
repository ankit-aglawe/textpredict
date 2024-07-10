import logging

from transformers import AutoTokenizer, Trainer, TrainingArguments

from textpredict.config import default_training_config
from textpredict.evaluation import compute_metrics, log_metrics
from textpredict.utils.hyperparameter_tuning import tune_hyperparameters

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        model_name=None,
        output_dir="./results",
        config=None,
        device="cpu",
        training_config=None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.config = config or {}
        self.training_config = {**default_training_config, **(training_config or {})}
        self.model = self.load_model(model_name)
        self.tokenizer = self.load_tokenizer(model_name)
        self.model.to(device)
        self.callbacks = []
        self.best_params = None
        self.state = None
        self.train_dataset = None  # To be set directly by the user
        self.val_dataset = None  # To be set directly by the user if needed
        logger.info(f"Trainer initialized with model {model_name} on {device}")

    def load_model(self, model_name):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def preprocess_data(self):
        pass

    def train(self, from_checkpoint=False):
        try:
            self.training_config["output_dir"] = self.output_dir

            training_args = TrainingArguments(**self.training_config)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
                callbacks=self.callbacks if self.callbacks else None,
            )
            if from_checkpoint:
                trainer.train(
                    resume_from_checkpoint=True
                )  # TODO: add path to take checkpoints from
            else:
                trainer.train()
            self.state = trainer.state
            log_metrics(trainer.state.log_history)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save(self):
        try:
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Model saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def get_metrics(self):
        assert self.state is not None, "Training state is not available."
        metrics = {}
        for log in self.state.log_history:
            for key, value in log.items():
                if key in metrics:
                    metrics[key].append(value)
                else:
                    metrics[key] = [value]
        return metrics

    def tune_hyperparameters(self, train_data, eval_data, n_trials=50):
        self.best_params = tune_hyperparameters(
            train_data=train_data,
            eval_data=eval_data,
            model_name=self.model_name,
            output_dir=self.output_dir,
            device=self.device,
            n_trials=n_trials,
        )
        return self.best_params

    def enable_logging(self, tool="wandb", project_name=None):
        if tool == "wandb":
            import wandb  # type: ignore

            wandb.init(project=project_name)

    def add_callback(self, callback):
        self.callbacks.append(callback)
