"""Load static demographics table."""

from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import DATA_PATH, load_dataset_from_file


def demographic_loader(
    table: str,
    demographic_col_name: str,
    nrows: Optional[int] = None,
    one_entry_per_patient: bool = True,
) -> pd.DataFrame:
    """Load static demographics table. Drops admissions without chartevents
    data and duplicate rows. Returns a df with columns for patient_id,
    timestamp, and value.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Static demographics table.
    """
    table_file_path = DATA_PATH / "mimic-iii-clinical-database-1.4" / table

    patients = load_dataset_from_file(
        file_path=table_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            demographic_col_name,
        ],
    )

    if one_entry_per_patient:

        # Keep only one entry per patient
        patients = patients.drop_duplicates(subset=["SUBJECT_ID"], keep="first")

    # Rename column
    patients = patients.rename(
        columns={"SUBJECT_ID": "patient_id"},
    )

    return patients.reset_index(drop=True)


@data_loaders.register("date_of_birth")
def load_dob(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load birthdays for calcalting age. Drops admissions without chartevents
    data and duplicate rows. Returns a df with columns for patient_id and
    value.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Birthdays.
    """

    dob = demographic_loader("PATIENTS.csv.gz", "DOB", nrows)

    # Rename column
    dob = dob.rename(columns={"DOB": "date_of_birth"})

    # Convert to datetime
    dob["date_of_birth"] = pd.to_datetime(dob["date_of_birth"], format="%Y-%m-%d")

    return dob.reset_index(drop=True)


@data_loaders.register("date_of_death")
def load_dod(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load date of deaths. Keeps only rows with dates and adds a value column with all 1's

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Date of deaths.
    """

    dod = demographic_loader("PATIENTS.csv.gz", "DOD", nrows)

    # rename column
    dod = dod.rename(columns={"DOD": "timestamp"})

    # drop rows with no date of death
    dod = dod.dropna()

    # convert to datetime
    dod["timestamp"] = pd.to_datetime(dod["timestamp"], format="%Y-%m-%d")

    # add a value column with all 1's
    dod["value"] = 1

    return dod.reset_index(drop=True)


@data_loaders.register("sex_is_female")
def load_sex(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load gender of each patient. Female patients are assigned a value of 1.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Dataframe with a column containing a gender for each patient.
    """

    gender = demographic_loader("PATIENTS.csv.gz", "GENDER", nrows)

    gender["GENDER"] = gender["GENDER"].map({"M": 0, "F": 1})

    # Rename column
    gender = gender.rename(columns={"GENDER": "sex_is_female"})

    return gender.reset_index(drop=True)


if __name__ == "__main__":
    load_dod(nrows=1000)
