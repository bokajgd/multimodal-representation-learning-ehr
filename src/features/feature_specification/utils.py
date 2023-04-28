"""Utils for specifying feature specs"""

import pickle as pkl
from pathlib import Path
from typing import Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEXT_MODEL_PATH = RELATIVE_PROJECT_ROOT / "text_models"

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
