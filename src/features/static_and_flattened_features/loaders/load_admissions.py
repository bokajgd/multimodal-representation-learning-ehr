"""Loaders for the admissions table."""
from typing import Optional

import numpy as np
import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import DATA_PATH, load_dataset_from_file


@data_loaders.register("admissions")
def load_admissions_base_df(
    nrows: Optional[int] = None,
    return_value_as_los: bool = False,
) -> pd.DataFrame:
    """Load admissions table. Drops admissions without chartevents data and
    duplicate rows. Returns a df with columns for patient_id, timestamp,
    admission_type, and value.

    Args:
        nrows (int): Number of rows to load.
        return_value_as_los (bool): If True, returns the length of the
            stay in days. If False, it returns 1 for all admissions. Defaults to False.

    Returns:
        pd.DataFrame: Admissions table.
    """
    admissions_file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "ADMISSIONS.csv.gz"
    )

    admissions = load_dataset_from_file(
        file_path=admissions_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "ADMISSION_TYPE",
            "HAS_CHARTEVENTS_DATA",
        ],
    )

    # Dropping admissions without chartevents data
    admissions = admissions[admissions["HAS_CHARTEVENTS_DATA"] == 1].drop(
        columns=["HAS_CHARTEVENTS_DATA"],
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
    if return_value_as_los:
        admissions["value"] = (
            admissions["DISCHTIME"] - admissions["ADMITTIME"]
        ).dt.total_seconds() / 86400
    else:
        admissions["value"] = 1

    # Renaming columns
    admissions = admissions.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "HADM_ID": "admission_id",
            "ADMITTIME": "admission_timestamp",
            "DISCHTIME": "timestamp",
            "ADMISSION_TYPE": "admission_type",
        },
    )

    return admissions


@data_loaders.register("admission_discharge_timestamps")
def load_admission_discharge_timestamps(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load admissions discharge times with admission_id column for merging.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Admissions discharge times with admission_id column.
    """
    admissions = load_admissions_base_df(nrows=nrows)

    return admissions[["admission_id", "timestamp"]]


@data_loaders.register("admission_timestamps")
def load_admission_timestamps(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load admissions times with admission_id column for merging.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Admissions discharge times with admission_id column.
    """
    admissions = load_admissions_base_df(nrows=nrows)

    return admissions[["admission_id", "admission_timestamp"]]


@data_loaders.register("all_admissions")
def load_all_admissions(
    nrows: Optional[int] = None,
    return_value_as_los: bool = True,
    timestamps_only: bool = False,
) -> pd.DataFrame:
    """Load all admissions for feature generation.

    Args:
        nrows (int): Number of rows to load.
        return_value_as_admission_length_days (bool): If True, returns the length of the
            admission in days instead. If not it returns 1 for all admissions. Defaults to True.
        timestamps_only (bool): If True, returns only the timestamps and patient_id.
            If not it also returns the value column Defaults to False.

    Returns:
        pd.DataFrame: Emergency admissions table.
    """

    all_admissions = load_admissions_base_df(
        nrows=nrows,
        return_value_as_los=return_value_as_los,
    ).drop(
        columns=["admission_type", "admission_id", "admission_timestamp"],
    )

    if timestamps_only:
        return all_admissions.drop(columns="value").reset_index(drop=True)
    else:
        return all_admissions.reset_index(drop=True)


@data_loaders.register("emergency_admissions")
def load_emergency_admissions(
    nrows: Optional[int] = None,
    return_value_as_los: bool = True,
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
    admissions = load_admissions_base_df(
        nrows=nrows,
        return_value_as_los=return_value_as_los,
    )

    emergency_admissions = admissions[admissions["admission_type"] == "EMERGENCY"].drop(
        columns=["admission_type", "admission_id", "admission_timestamp"],
    )

    if timestamps_only:
        return emergency_admissions.drop(columns="value").reset_index(drop=True)
    else:
        return emergency_admissions.reset_index(drop=True)


def _load_services(nrows):
    file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "SERVICES.csv.gz"
    )

    services = load_dataset_from_file(
        file_path=file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "HADM_ID",
            "TRANSFERTIME",
            "CURR_SERVICE",
        ],
    )
    
    admissions = load_admissions_base_df(nrows=nrows)[["admission_id", "admission_type"]]

    # rename columns
    services = services.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "HADM_ID": "admission_id",
            "TRANSFERTIME": "timestamp",
        },
    )
        
    # merge admission_type onto services
    services = services.merge(admissions, on="admission_id")

    # convert timestamp to datetime
    services["timestamp"] = pd.to_datetime(services["timestamp"])

    # remove newborn admissions
    services = services[services["admission_type"] != "NEWBORN"]

    return services.reset_index(drop=True)


@data_loaders.register("scheduled_surgical")
def load_scheduled_surgical(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load scheduled surgical admissions table."""
    
    services = _load_services(nrows=nrows)
    
    # flag elective surgical services as 1, the rest as 0
    services["value"] = np.where(
        (services["CURR_SERVICE"].str.contains("SURG")) & (services["admission_type"] == "ELECTIVE"),
        1,
        0,
    )

    return services[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


@data_loaders.register("unscheduled_surgical")
def load_unscheduled_surgical(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load unscheduled surgical admissions table."""
    
    services = _load_services(nrows=nrows)
    
    # flag emergency surgical services as 1, the rest as 0
    services["value"] = np.where(
        (services["CURR_SERVICE"].str.contains("SURG")) & (services["admission_type"].isin(['EMERGENCY', 'URGENT'])),
        1,
        0,
    )

    return services[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


@data_loaders.register("medical")
def load_medical(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load medical admissions table."""
    
    services = _load_services(nrows=nrows)
    
    # flag medical services as 1, the rest as 0
    services["value"] = np.where(
        (~services["CURR_SERVICE"].str.contains("SURG")),
        1,
        0,
    )

    return services[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


if __name__ == "__main__":
    df = load_medical()
