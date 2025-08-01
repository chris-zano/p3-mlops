import pytest
from unittest.mock import patch, MagicMock
from scripts.train import _load_dataset, load_tokenizer, tokenize_function, write_model_version_to_secrets_manager



def test_load_dataset():
    dataset = _load_dataset("tmdb/tmdb-movie-metadata")
    assert "train" in dataset
    assert "validation" in dataset
    assert len(dataset["train"]) > 0

def test_load_tokenizer():
    tokenizer = load_tokenizer("google/flan-t5-base")
    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")

def test_tokenize_function():
    load_tokenizer("google/flan-t5-base")
    sample = {"description": ["A hero saves the world"], "title": ["Superhero"]}
    result = tokenize_function(sample)
    assert "input_ids" in result
    assert "labels" in result

@patch("scripts.train.boto3.client")
def test_write_model_version_to_secrets_manager(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client

    write_model_version_to_secrets_manager("my-secret", 42)

    mock_client.update_secret.assert_called_once()
    args, kwargs = mock_client.update_secret.call_args
    assert kwargs["SecretId"] == "my-secret"
    assert '"latest_model_version": 42' in kwargs["SecretString"]