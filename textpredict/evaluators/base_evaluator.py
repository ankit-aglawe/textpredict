from transformers import AutoTokenizer, Trainer, TrainingArguments

from textpredict.config import default_evaluation_config
from textpredict.device_manager import DeviceManager
from textpredict.evaluation import compute_metrics, log_metrics
from textpredict.logger import get_logger

logger = get_logger(__name__)


class BaseEvaluator:
    def __init__(
        self, model_name=None, config=None, device=None, evaluation_config=None
    ):
        self.model_name = model_name

        self.device = device or DeviceManager.get_device()

        self.config = config or {}
        self.evaluation_config = {
            **default_evaluation_config,
            **(evaluation_config or {}),
        }
        self.model = self.load_model(model_name, self.device)
        self.tokenizer = self.load_tokenizer(model_name)
        self.model.to(device)
        self.data = None  # To be set directly by the user
        logger.info(f"Evaluator initialized with model {model_name} on {device}")

    def load_model(self, model_name):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def preprocess_data(self):
        pass

    def evaluate(self):
        try:
            self.evaluation_config["output_dir"] = self.evaluation_config.get(
                "output_dir", "./results"
            )

            evaluation_args = TrainingArguments(**self.evaluation_config)

            trainer = Trainer(
                model=self.model,
                args=evaluation_args,
                eval_dataset=self.data,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
            )
            metrics = trainer.evaluate()
            log_metrics(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
