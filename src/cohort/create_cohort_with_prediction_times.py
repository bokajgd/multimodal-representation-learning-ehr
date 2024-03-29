"""Create cohort with prediction times."""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders
from utils import DATA_PATH, load_dataset_from_file, load_sql_query

ICU_STAYS_QUERY = """
        SELECT icu.SUBJECT_ID, icu.OUTTIME, icu.INTIME, icu.LOS, icu.HADM_ID,
        FROM physionet-data.mimiciii_clinical.icustays icu
        WHERE icu.HADM_ID IN (
            SELECT DISTINCT ic.HADM_ID
            FROM physionet-data.mimiciii_clinical.icustays ic
            JOIN physionet-data.mimiciii_clinical.admissions adm
            ON ic.HADM_ID = adm.HADM_ID
            WHERE ic.DBSOURCE = 'metavision'
            AND adm.HAS_CHARTEVENTS_DATA = 1
        )
        """

PATIENTS_QUERY = """
        SELECT pat.SUBJECT_ID, pat.DOB,
        FROM physionet-data.mimiciii_clinical.patients pat
        """


@data_loaders.register("prediction_times")
def generate_prediction_times_df(
    nrows: Optional[int] = None,
    save_to_disk: bool = False,
) -> pd.DataFrame:
    """Load icu stays table to generate prediction times. Drops admissions
    without chartevents data, patients below the age of 15, icu stays with LOS
    under 24 hours and duplicate rows. Returns a df with columns for patient_id
    and timestamp. Prediction times are the time of admission to the ICU + 48
    hours.

    Args:
        nrows (int): Number of rows to load.
        save_to_disk (bool): If True, saves the cohort to disk. Defaults to False.

    Returns:
        pd.DataFrame: Prediction times df
    """
    cohort_output_file_path = DATA_PATH / "misc" / "cohort_with_prediction_times.csv"

    cohort = load_sql_query(ICU_STAYS_QUERY)
    patients = load_sql_query(PATIENTS_QUERY)

    # Convert to datetime
    cohort["OUTTIME_date"] = pd.to_datetime(cohort["OUTTIME"].copy()).dt.date
    patients["DOB"] = pd.to_datetime(patients["DOB"]).dt.date

    # Merge the DOB column to the cohort using the SUBJECT_ID column as the key
    cohort = cohort.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID")

    # Drop rows with a null value in the DOB or OUTTIME column
    cohort = cohort.dropna(subset=["DOB", "OUTTIME_date"]).reset_index(drop=True)

    # Calculate age for each patient
    cohort["AGE"] = cohort.apply(
        lambda d: round((d["OUTTIME_date"] - d["DOB"]).days / 365.25, 2),
        axis=1,
    )

    # Drop patients below the age of 15, drop rows with a null value, drop duplicates and drop rows with a LOS under 1
    cohort = (
        cohort.query("AGE >= 15")
        .query("LOS >= 1")
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Drop all columns except for SUBJECT_ID and OUTTIME
    cohort = cohort.drop(
        cohort.columns.difference(["SUBJECT_ID", "INTIME", "HADM_ID"]),
        axis=1,
    )

    # Add 24 hours to the intime column values
    cohort["INTIME"] = cohort["INTIME"] + pd.Timedelta(hours=24)

    # Rename columns
    cohort = cohort.rename(
        columns={"SUBJECT_ID": "patient_id", "INTIME": "timestamp"},
    )

    if save_to_disk:
        cohort.to_csv(cohort_output_file_path, index=False)

    else:
        return cohort.reset_index(drop=True)


if __name__ == "__main__":
    generate_prediction_times_df(save_to_disk=True)
