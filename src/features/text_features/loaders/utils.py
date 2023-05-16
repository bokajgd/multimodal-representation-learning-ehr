"""A set of misc.

utility functions for free-text notes loaders.
"""
from typing import Optional
import re
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = RELATIVE_PROJECT_ROOT / "data"


def load_sql_query(query: str) -> str:
    client = bigquery.Client()

    return client.query(query).to_dataframe()


def text_preprocessing(
    df: pd.DataFrame,
    text_column_name: str = "text",
) -> pd.DataFrame:
    """Preprocess texts by lower casing, removing stopwords and symbols.

    Args:
        df (pd.DataFrame): Dataframe with a column containing text to clean.
        text_column_name (str): Name of column containing text. Defaults to "value".
    Returns:
        pd.DataFrame: _description_
    """
    regex_symbol_removal = re.compile(
        r"[^A-Za-z ]+|\b%s\b" % r"\b|\b",
    )

    df[text_column_name] = (
        df[text_column_name]
        .str.lower()
        .replace(regex_symbol_removal, value="", regex=True)
    )

    return df


def load_dataset_from_file(
    file_path: Path,
    nrows: Optional[int] = None,
    cols_to_load: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load dataset from file. Handles csv and parquet files based on suffix.

    Args:
        file_path (str): File name.
        nrows (int): Number of rows to load.
        cols_to_load (list[str]): Columns to load.

    Returns:
        pd.DataFrame: Dataset
    """

    file_suffix = file_path.suffix

    if file_suffix == ".csv":
        if cols_to_load:
            return pd.read_csv(file_path, nrows=nrows, usecols=cols_to_load)
        else:
            return pd.read_csv(file_path, nrows=nrows)
    elif file_suffix == ".gz":
        if cols_to_load:
            return pd.read_csv(
                file_path,
                compression="gzip",
                nrows=nrows,
                usecols=cols_to_load,
            )
        else:
            return pd.read_csv(file_path, compression="gzip", nrows=nrows)
    else:
        raise ValueError(f"Invalid file suffix {file_suffix}")
