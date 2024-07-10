from textpredict.model_loader import load_model, load_model_from_directory


def test_load_model():
    model = load_model(
        model_name="distilbert-base-uncased-finetuned-sst-2-english", task="sentiment"
    )
    assert model is not None
    assert model.model_name == "distilbert-base-uncased-finetuned-sst-2-english"


def test_load_model_from_directory(tmp_path):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    # Mock the necessary files
    (model_dir / "pytorch_model.bin").touch()
    config_file = model_dir / "config.json"
    vocab_file = model_dir / "vocab.txt"

    with config_file.open("w") as f:
        f.write('{"model_type": "distilbert"}')

    with vocab_file.open("w") as f:
        f.write("")

    model = load_model_from_directory(model_dir, task="sentiment")
    assert model is not None
    assert model.model_name == str(model_dir)
