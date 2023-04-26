"""A set of misc. utility functions for free-text notes loaders.
"""
import re

import pandas as pd
from pathlib import Path

from google.cloud import bigquery

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = RELATIVE_PROJECT_ROOT / "data"

def load_sql_query(query: str) -> str:
    client = bigquery.Client()

    return client.query(query).to_dataframe()


def text_preprocessing(
    df: pd.DataFrame,
    text_column_name: str = "value",
) -> pd.DataFrame:
    """Preprocess texts by lower casing, removing stopwords and symbols.
    Args:
        df (pd.DataFrame): Dataframe with a column containing text to clean.
        text_column_name (str): Name of column containing text. Defaults to "value".
    Returns:
        pd.DataFrame: _description_
    """
    regex_symbol_removal = re.compile(
        r"[^A-Za-z ]+|\b%s\b" % r"\b|\b")

    df[text_column_name] = (
        df[text_column_name]
        .str.lower()
        .replace(regex_symbol_removal, value="", regex=True)
    )

    return df
