"""Function for fitting text model.

Code adapted from GitHub repository 'psycop-feature-generation'
"""

from typing import Literal, Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def fit_text_model(
    model: Literal["bow", "tfidf"],
    df: pd.DataFrame,
    text_column_name: str = "value",
    ngram_range: tuple = (1, 1),
    max_df: float = 1.0,
    min_df: int = 1,
    max_features: Optional[int] = 100,
) -> Union[CountVectorizer, TfidfVectorizer]:
    """Fits a bag-of words model on a corpus
    Args:
        model (Literal[str]): Which model to use. Takes either "bow" or "tfidf".
        df (pd.DataFrame): Dataframe with text column to fit model on.
        text_column_name (str): Name of column containing text. Defaults to "value".
        ngram_range (tuple, optional): The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams. Defaults to (1, 1).
        max_df (float, optional): The proportion of documents the words should appear in to be included. Defaults to 0.95.
        min_df (int, optional): Remove words occuring in less than min_df documents. Defaults to 2.
        max_features (int | None, optional): If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used. Defaults to 500.
    Returns:
        Union[CountVectorizer, TfidfVectorizer]: Fitted model
    """

    vec_type = {"bow": CountVectorizer, "tfidf": TfidfVectorizer}

    if model not in vec_type.keys():
        raise ValueError(
            f"model name '{model}' not in vec_type. Available choices are 'bow' or 'tfidf'",
        )

    # Define vectorizer
    vec = vec_type[model](
        stop_words="english",
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )

    # Fit to corpus
    vec.fit(df[text_column_name].values)

    return vec
