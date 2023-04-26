"""Pipeline for fitting text models

Code adapted from GitHub repository 'psycop-feature-generation'
"""

import logging
from collections.abc import Sequence
from pathlib import Path
import pandas as pd
from typing import Any, Literal, Optional
from utils.fit_text_model import fit_text_model

from utils.utils import save_text_model_to_dir, save_model_vocab_to_dir, TEXT_MODEL_PATH

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def create_model_filename(
    model: Literal["bow", "tfidf"],
    df: pd.DataFrame,
    ngram_range: tuple,
    max_df: float,
    min_df: int,
    max_features: Optional[int],
):
    """Create model filename including all relevant informaiton about the model.
    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        df (pd.DataFrame): Dataframe with text column to fit model on.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
        max_df (float): The proportion of documents the words should appear in to be included.
        min_df (int): Remove words occuring in less than min_df documents.
        max_features (int, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used.
    """
    max_df_str = str(max_df).replace(".", "")
    ngram_range_str = "".join(c for c in str(ngram_range) if c.isdigit())

    return f"{model}_ngram_range_{ngram_range_str}_max_df_{max_df_str}_min_df_{min_df}_max_features_{max_features}.pkl"


def text_model_pipeline(
    model: Literal["bow", "tfidf"],
    df: pd.DataFrame,
    text_column_name: str = "value",
    ngram_range: tuple = (1, 1),
    max_df: float = 1.0,
    min_df: int = 1,
    max_features: Optional[int] = 100,
    save_path: str = TEXT_MODEL_PATH,
) -> Any:
    """Pipeline for fitting and saving a bag-of-words or tfidf model
    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        df (pd.DataFrame): Dataframe with text column to fit model on.
        text_column_name (str): Name of column containing text. Defaults to "value".
        n_rows (int, optional): How many rows to include in the loaded data. If None, all are included. Defaults to None.
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 2).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
        save_path (str, optional): Path where the model will be saved. Defaults to "E:/shared_resources/text_models".
    Returns:
        str: Log info on the path and filename of the fitted text model.
    """
    # create model filename from params
    filename = create_model_filename(
        model=model,
        df=df,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )


    # fit model
    vec = fit_text_model(
        model=model,
        df=df,
        text_column_name=text_column_name,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # save model and vocab to dir
    save_text_model_to_dir(model=vec, filename=filename, save_path=save_path)
    save_model_vocab_to_dir(model=vec, model_filename=filename, save_path=save_path, )

    return None