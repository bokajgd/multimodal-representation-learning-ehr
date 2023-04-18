"""Loaders for diagnoses tables."""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders
from .utils import load_dataset_from_file, DATA_PATH
from .load_admissions import load_admission_discharge_timestamps

def _generate_icd9_range(icd9_code_interval: tuple[int,int]) -> tuple[str,str]:
    """Generate ICD9 strings for a range of ICD9 codes. 

    Args:
        icd9_code_interval (tuple[int,int]): Tuple of start and end ICD9 codes.

    Returns:
        tuple[str,str]: Tuple of start and end ICD9 codes as strings.
    """

    # Define range of codes
    icd9_range = range(icd9_code_interval[0], icd9_code_interval[1] + 1)

    # Convert range to strings with leading zeros
    icd9_codes = [f"{code:03d}" for code in icd9_range]

    return tuple(icd9_codes)


@data_loaders.register("all_diagnoses")
def load_diagnoses(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load diagnoses dataframe with all diagnoses for each admission. Merge admission_disharge_timestamps
      with diagnoses to get the timestamp for each diagnosis using admission_ids.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Diagnoses table.
    """
    diagnoses_file_path = DATA_PATH / "mimic-iii-clinical-database-1.4" / "DIAGNOSES_ICD.csv.gz"

    diagnoses = load_dataset_from_file(
        file_path=diagnoses_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "HADM_ID",
            "SEQ_NUM",
            "ICD9_CODE",
        ],
    )

    diagnoses = diagnoses.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "HADM_ID": "admission_id",
        }
    )

    admission_discharge_timestamps = load_admission_discharge_timestamps(nrows=nrows)

    # Merge admission timestamps
    diagnoses = pd.merge(diagnoses, admission_discharge_timestamps, on="admission_id")

    return diagnoses.reset_index(drop=True)


@data_loaders.register("a_diagnoses")
def load_a_diagnoses(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all A-diagnoses for each admission."""

    diagnoses = load_diagnoses(nrows=nrows)
    
    return diagnoses[diagnoses["SEQ_NUM"] == 1]


@data_loaders.register("tuberculosis_a_diagnoses")
def load_tuberculosis_a_diagnosis(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all tuberculosis A-diagnoses for each admission."""

    a_diagnoses = load_a_diagnoses(nrows=nrows)

    tuberculosis_codes = _generate_icd9_range((100, 180))

    tuberculosis = a_diagnoses.loc[a_diagnoses["ICD9_CODE"].str.startswith(tuberculosis_codes)]

    # Add value col for feature generation
    tuberculosis["value"] = 1

    return tuberculosis[["patient_id", "timestamp", "value"]].reset_index(drop=True)


if __name__ == "__main__":
    diagnoses = load_diagnoses()
    tuberculosis = load_tuberculosis_a_diagnosis()
    print('imported')