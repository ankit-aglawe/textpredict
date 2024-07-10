from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def load_data(
    dataset_name=None,
    data_files=None,
    dataset=None,
    tokenizer_name="bert-base-uncased",
    splits=["train"],
):
    """
    Load and preprocess the dataset.

    Args:
        dataset_name (str, optional): The name of the dataset to load from Hugging Face. Defaults to None.
        data_files (dict, optional): A dictionary with paths to the dataset files. Defaults to None.
        dataset (Dataset or DatasetDict, optional): A pre-loaded Hugging Face dataset. Defaults to None.
        tokenizer_name (str, optional): The name of the tokenizer to use. Defaults to "bert-base-uncased".
        splits (list, optional): List of dataset splits to load. Defaults to ["train"].

    Returns:
        dict: A dictionary with preprocessed datasets for each split.
    """
    if dataset is not None:
        if isinstance(dataset, DatasetDict):
            loaded_datasets = {split: dataset[split] for split in splits}

        elif isinstance(dataset, Dataset):
            if len(splits) > 1:
                raise ValueError(
                    "Provided dataset is a single split but multiple splits were requested."
                )
            loaded_datasets = {splits[0]: dataset}

        else:
            raise ValueError(
                "Provided dataset must be a Hugging Face Dataset or DatasetDict object."
            )

    elif dataset_name is not None or data_files is not None:
        loaded_datasets = {
            split: load_dataset(dataset_name, data_files=data_files, split=split)
            for split in splits
        }

    else:
        raise ValueError(
            "Either 'dataset_name', 'data_files', or 'dataset' must be provided."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = {
        split: ds.map(preprocess_function, batched=True)
        for split, ds in loaded_datasets.items()
    }

    for split in tokenized_datasets:
        tokenized_datasets[split].set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

    return tokenized_datasets
