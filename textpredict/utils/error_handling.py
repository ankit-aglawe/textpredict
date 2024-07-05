# textpredict/utils/error_handling.py

import logging

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Custom exception for model-related errors."""

    pass


class DataError(Exception):
    """Custom exception for data-related errors."""

    pass


def log_and_raise(exception, message):
    """
    Log an error message and raise the specified exception.

    Args:
        exception (Exception): The exception class to raise.
        message (str): The error message to log and raise.

    Raises:
        exception: The specified exception with the provided message.
    """
    logger.error(message)
    raise exception(message)
