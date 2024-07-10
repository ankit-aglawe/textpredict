import logging
import re

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean the input text by removing unwanted characters.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    try:
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", "", text)
        text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise


def tokenize_text(text: str, tokenizer) -> list:
    """
    Tokenize the input text using the provided tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer: The tokenizer to use for tokenization.

    Returns:
        list: The tokenized text.
    """
    try:
        return tokenizer.tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        raise


def preprocess_text(text: str, tokenizer) -> list:
    """
    Preprocess the input text by cleaning and tokenizing it.

    Args:
        text (str): The input text to preprocess.
        tokenizer: The tokenizer to use for tokenization.

    Returns:
        list: The preprocessed (tokenized) text.
    """
    try:
        cleaned_text = clean_text(text)
        return tokenize_text(cleaned_text, tokenizer)
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise
