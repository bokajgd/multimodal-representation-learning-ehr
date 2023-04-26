"""Utils for handling, processing and model fitting free-text notes."""

import pickle as pkl
from pathlib import Path
from typing import Any, Union

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEXT_MODEL_PATH = RELATIVE_PROJECT_ROOT / "text_models"


def save_text_model_to_dir(
    model: Any,
    filename: str,
    save_path: str = TEXT_MODEL_PATH,
):
    """
    Saves the model to a pickle file
    Args:
        model (Any): The model to save
        filename (str): The filename to save the model as
        save_path (str): The path where the model will be saved
    """

    filepath = Path(save_path) / filename

    with Path(filepath).open("wb") as f:
        pkl.dump(model, f)


def save_model_vocab_to_dir(
    model: Any,
    model_filename: str,
    save_path: str = TEXT_MODEL_PATH,
):

    # Saving text model vocabulary
    vocab_path = Path(save_path) / "vocabs" / f"{model_filename}_vocab.pkl"
    with open(vocab_path, "wb") as fw:
        joblib.dump(model.vocabulary_, fw)


def load_text_model(
    filename: str,
    path_str: str = TEXT_MODEL_PATH,
) -> Union[CountVectorizer, TfidfVectorizer]:
    """
    Loads a text model from a pickle file
    Args:
        filename: filename name of the model
        path_str: path of model location
    """

    filepath = Path(path_str) / filename

    with Path(filepath).open("rb") as f:
        return pkl.load(f)
