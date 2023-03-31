"""Create cohort with prediction times."""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders
from utils import DATA_PATH, load_dataset_from_file


@data_loaders.register("prediction_times")
def generate_prediction_times_df(
    nrows: Optional[int] = None,
    save_to_disk: bool = False,
) -> pd.DataFrame:
    """Load icu stays table to generate prediction times. Drops admissions without chartevents data,
    patients below the age of 18, icu stays under 48 hours and duplicate rows. Returns a df with columns for patient_id and timestamp.
    Prediction times are the time of admission to the ICU + 48 hours.

    Args:
        nrows (int): Number of rows to load.
        save_to_disk (bool): If True, saves the cohort to disk. Defaults to False.

    Returns:
        pd.DataFrame: Prediction times df
    """
    icu_stays_file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "ICUSTAYS.csv.gz"
    )

    patients_file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "PATIENTS.csv.gz"
    )

    cohort_output_file_path = (
        DATA_PATH / "misc" / "cohort_with_prediction_times.csv"
    )

    cohort = load_dataset_from_file(
        file_path=icu_stays_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "OUTTIME",
            "LOS",
        ],
    )

    patients = load_dataset_from_file(
        file_path=patients_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "DOB",
        ],
    )

    # Convert to datetime
    cohort["OUTTIME_date"] = pd.to_datetime(cohort["OUTTIME"].copy()).dt.date
    patients["DOB"] = pd.to_datetime(patients["DOB"]).dt.date

    # Merge the DOB column to the cohort using the SUBJECT_ID column as the key
    cohort = cohort.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID") 

    # Drop rows with a null value in the DOB or OUTTIME column
    cohort = cohort.dropna(subset=["DOB", "OUTTIME_date"]).dropna().reset_index(drop=True)

    # Calculate age for each patient
    cohort['AGE'] = cohort.apply(lambda d: round((d['OUTTIME_date'] - d['DOB']).days / 365.25 , 2), axis=1)

    # Drop patients below the age of 18, drop rows with a null value, drop duplicates and drop rows with a LOS under 2
    cohort = cohort.query('AGE >= 18').query('LOS >= 2').drop_duplicates().reset_index(drop=True)

    # Drop all columns except for SUBJECT_ID and OUTTIME
    cohort = cohort.drop(cohort.columns.difference(["SUBJECT_ID", "OUTTIME"]), axis=1)

    # Rename columns
    cohort = cohort.rename(
        columns={"SUBJECT_ID": "patient_id", "OUTTIME": "timestamp",},
    )

    if save_to_disk:
        cohort.to_csv(cohort_output_file_path, index=False)
    
    else:
        return cohort.reset_index(drop=True)


if __name__ == "__main__":
    generate_prediction_times_df()

