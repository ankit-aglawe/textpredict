import pytest

from textpredict.trainers.sequence_classification_trainer import (
    SequenceClassificationTrainer,
)


@pytest.fixture
def trainer():
    return SequenceClassificationTrainer(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        output_dir="./results",
    )


def test_load_data(trainer):
    trainer.load_data(dataset_name="imdb", split="train")
    assert trainer.dataset is not None


def test_preprocess_data(trainer):
    trainer.load_data(dataset_name="imdb", split="train")
    trainer.preprocess_data(
        tokenizer_name="distilbert-base-uncased-finetuned-sst-2-english"
    )
    assert "input_ids" in trainer.dataset.column_names
