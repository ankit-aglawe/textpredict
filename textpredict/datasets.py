from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def load_data(
    dataset_name=None,
    data_files=None,
    dataset=None,
    tokenizer_name="bert-base-uncased",
    splits=["train"],
    text_column=None,
    label_column=None,
):
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
        model_inputs = tokenizer(
            examples[text_column], padding="max_length", truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[label_column], padding="max_length", truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        return model_inputs

    tokenized_datasets = {
        split: ds.map(preprocess_function, batched=True)
        for split, ds in loaded_datasets.items()
    }

    for split in tokenized_datasets:
        tokenized_datasets[split].set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
            ],
        )

    return tokenized_datasets
