# textpredict/datasets.py

import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_data(dataset_name: str, split: str = "train"):
    """
    Load a dataset split.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The dataset split to load (e.g., 'train', 'test'). Defaults to 'train'.

    Returns:
        Dataset: The loaded dataset split.
    """
    try:
        logger.info(f"Loading {dataset_name} dataset with split {split}")
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise


def get_dataset_splits(dataset_name: str):
    """
    Get the available splits for a dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        list: A list of available splits for the dataset.
    """
    try:
        logger.info(f"Getting splits for {dataset_name} dataset")
        dataset_info = load_dataset(dataset_name)
        return list(dataset_info.keys())
    except Exception as e:
        logger.error(f"Error getting splits for dataset {dataset_name}: {e}")
        raise
