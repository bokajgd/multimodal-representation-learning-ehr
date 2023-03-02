"""A set of misc. utility functions for data loaders. """

from pathlib import Path
from typing import Any, Optional

import catalogue
import pandas as pd

RELATIVE_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = RELATIVE_PROJECT_ROOT / "data"


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

    elif file_suffix == ".parquet":
        if nrows:
            raise ValueError("nrows not supported for parquet files")

        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Invalid file suffix {file_suffix}")


def write_df_to_file(
    df: pd.DataFrame,
    file_path: Path,
):
    """Write dataset to file. Handles csv and parquet files based on suffix.

    Args:
        df: Dataset
        file_path (str): File name.
    """

    file_suffix = file_path.suffix

    if file_suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif file_suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Invalid file suffix {file_suffix}")