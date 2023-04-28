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
