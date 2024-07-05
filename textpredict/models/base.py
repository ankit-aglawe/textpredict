from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from textpredict.logger import get_logger

logger = get_logger(__name__)


class BaseModel:
    def __init__(
        self,
        model_name: str,
        task: str,
        model=None,
        tokenizer=None,
        multi_label: bool = False,
    ):
        """
        Initialize the base model with the specified parameters.

        Args:
            model_name (str): The name of the model to load.
            task (str): The task type (e.g., 'sentiment-analysis').
            multi_label (bool): Whether the model supports multi-label classification.
        """
        try:
            self.model_name = model_name
            self.task = task
            self.multi_label = multi_label

            logger.info(f"Initializing {task} model: {model_name}")

            if model and tokenizer:
                self.model = model
                self.tokenizer = tokenizer
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.pipeline = pipeline(
                    task, model=self.model, tokenizer=self.tokenizer
                )

            # GPU setup
            # if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            #     self.device = torch.device("cuda")
            # elif (
            #     hasattr(torch.backends, "mps")
            #     and torch.backends.mps.is_available()
            #     and torch.backends.mps.is_built()
            # ):
            #     self.device = torch.device("mps")
            # else:
            #     self.device = torch.device("cpu")

            # self.parallel = torch.cuda.device_count() > 1
            # if self.parallel:
            #     self.model = torch.nn.DataParallel(self.model)
            # self.model.to(self.device)

            # logger.info(f"Model {model_name} loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            raise

    def predict(self, text: str or list, class_list: list = None) -> list:  # type: ignore
        """
        Make predictions using the model.

        Args:
            text (str or list): The input text or list of texts to classify.
            class_list (list, optional): The list of candidate labels for zero-shot classification.

        Returns:
            list: The prediction results.
        """
        try:
            if self.multi_label and class_list is not None:
                return self.pipeline(text, candidate_labels=class_list)
            return self.pipeline(text)
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
