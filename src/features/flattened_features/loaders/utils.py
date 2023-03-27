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
    elif file_suffix == ".gz":
        if cols_to_load:
            return pd.read_csv(file_path, compression='gzip', nrows=nrows, usecols=cols_to_load)
        else:
            return pd.read_csv(file_path, compression='gzip', nrows=nrows)
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
    
    
def _drop_rows_with_too_small_value_frequency(
    df: pd.DataFrame, 
    value_col_name: str, 
    threshold: float,
) -> pd.DataFrame:
    """Drop rows if the value in a given column only appears in less than n% of the rows.
    
    Args:
        df (pd.DataFrame): Dataframe to drop rows from.
        value_col_name (str): Name of column to check.
        threshold (float): Threshold for dropping rows.
    
    Returns:
        pd.DataFrame: Dataframe with rows dropped.
    """

    value_frequency = df[f'{value_col_name}'].value_counts(normalize=True)

    df = df.loc[df[f'{value_col_name}'].isin(value_frequency[value_frequency >= threshold].index)]

    return df.reset_index(drop=True)


def _drop_rows_with_too_small_patient_or_admission_frequency(
    df: pd.DataFrame,
    patient_or_admission_col_name: str,
    item_col_name: str,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Drop rows if an item (e.g. diagnosis, fluid, medication etc.) only appears in less than n% of the patients/admissions.
    
    Args:
        df (pd.DataFrame): Dataframe to drop rows from.
        patient_or_admission_col_name (str): Name of column to group by.
        item_col_name (str): Name of column to check.
        threshold (float): Threshold for dropping rows.
    
    Returns:
        pd.DataFrame: Dataframe with rows dropped.
    """

    # Calculate the number of unique patients or admissions
    unique_patients_or_admissions = len(df[f"{patient_or_admission_col_name}"].unique())

    item_frequencies = df.groupby([f"{item_col_name}"])[f"{patient_or_admission_col_name}"].nunique() / unique_patients_or_admissions

    itmes_to_drop = item_frequencies[item_frequencies < threshold].index

    # Drop all rows with items that don't meet the threshold
    df = df[~df[f"{item_col_name}"].isin(itmes_to_drop)]

    return df.reset_index(drop=True)