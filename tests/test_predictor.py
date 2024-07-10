import pytest

from textpredict.predictor import TextPredict


@pytest.fixture
def text_predict():
    return TextPredict(device="cpu")


def test_initialize(text_predict):
    text_predict.set_task("sentiment")
    text_predict.initialize(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        task="sentiment",
        source="huggingface",
    )
    assert (
        text_predict.default_model_name
        == "distilbert-base-uncased-finetuned-sst-2-english"
    )
    assert text_predict.current_task == "sentiment"


def test_analyze(text_predict):
    text_predict.set_task("sentiment")
    text_predict.initialize(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        task="sentiment",
        source="huggingface",
    )
    result = text_predict.analyze(text="I love this product!")
    assert result is not None
    assert isinstance(result, list)
