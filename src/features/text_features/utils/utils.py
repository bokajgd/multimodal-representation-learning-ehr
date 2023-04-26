""" Utils for handling, processing and model fitting free-text notes  """

import pickle as pkl
from pathlib import Path
from typing import Any, Union

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