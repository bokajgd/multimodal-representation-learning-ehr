"""Loaders for the admissions table."""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders
from .utils import (load_dataset_from_file, DATA_PATH,)


@data_loaders.register("admissions")
def load_admissions(
    nrows: Optional[int] = None,
    return_value_as_admission_length_days: bool = False,
) -> pd.DataFrame:
    """Load admissions table. Drops admissions without chartevents data and duplicate rows. 
    Returns a df with columns for patient_id, timestamp, admission_type, and value.

    Args:
        nrows (int): Number of rows to load.
        return_value_as_admission_length_days (bool): If True, returns the length of the
            admission in days instead. If not it returns 1 for all admissions. Defaults to False.

    Returns:
        pd.DataFrame: Admissions table.
    """
    admissions_file_path = DATA_PATH / "raw" / "ADMISSIONS.csv"

    admissions = load_dataset_from_file(
        file_path=admissions_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "ADMITTIME",
            "DISCHTIME",
            "ADMISSION_TYPE",
            "HAS_CHARTEVENTS_DATA",
        ],
    )

    # Dropping admissions without chartevents data
    admissions = admissions[admissions["HAS_CHARTEVENTS_DATA"] == 1].drop(
        columns=["HAS_CHARTEVENTS_DATA"]
    )

    # Drop duplicates
    admissions.reset_index(drop=True).drop_duplicates(
        subset=["SUBJECT_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE"],
        keep="first",
    )

    # Convert to datetime
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"])

    # Add value column for feature generation
    if return_value_as_admission_length_days:
        admissions["value"] = (
            admissions["DISCHTIME"] - admissions["ADMITTIME"]
        ).dt.total_seconds() / 86400
    else:
        admissions["value"] = 1
    
    # Drop admission time
    admissions = admissions.drop(columns="ADMITTIME")

    # Renaming columns
    admissions = admissions.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "DISCHTIME": "timestamp",
            "ADMISSION_TYPE": "admission_type",
        }
    )

    return admissions.reset_index(drop=True)


@data_loaders.register("emergency_admissions")
def load_emergency_admissions(
    nrows: Optional[int] = 100,
    return_value_as_admission_length_days: bool = True,
    timestamps_only: bool = False,
) -> pd.DataFrame:
    """Load emergency admissions table.

    Args:
        nrows (int): Number of rows to load.
        return_value_as_admission_length_days (bool): If True, returns the length of the
            admission in days instead. If not it returns 1 for all admissions. Defaults to True.
        timestamps_only (bool): If True, returns only the timestamps and patient_id. 
            If not it also returns the value column Defaults to False.

    Returns:
        pd.DataFrame: Emergency admissions table.
    """
    admissions = load_admissions(nrows=nrows, return_value_as_admission_length_days = return_value_as_admission_length_days)

    emergency_admissions = admissions[admissions["admission_type"] == "EMERGENCY"].drop(
        columns=["admission_type"]
    )

    if timestamps_only:
        return emergency_admissions.drop(columns="value").reset_index(drop=True)
    else:
        return emergency_admissions.reset_index(drop=True)



if __name__ == "__main__":
    admissions = load_emergency_admissions(nrows=100)
