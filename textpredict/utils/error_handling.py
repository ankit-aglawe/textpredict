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


def safe_execute(func):
    """
    Decorator to wrap functions with error handling.

    Args:
        func (function): The function to wrap.

    Returns:
        function: The wrapped function with error handling.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataError as de:
            log_and_raise(DataError, f"Data error in {func.__name__}: {de}")
        except ModelError as me:
            log_and_raise(ModelError, f"Model error in {func.__name__}: {me}")
        except Exception as e:
            log_and_raise(Exception, f"Unexpected error in {func.__name__}: {e}")

    return wrapper
