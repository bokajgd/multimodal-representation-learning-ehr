"""Loaders for diagnoses tables."""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from load_admissions import load_admission_timestamps
from utils import load_sql_query

BASE_QUERY = """
        SELECT icd.SUBJECT_ID, icd.HADM_ID, icd.SEQ_NUM, icd.ICD9_CODE,
        FROM physionet-data.mimiciii_clinical.diagnoses_icd icd
        WHERE icd.HADM_ID IN (
            SELECT DISTINCT ic.HADM_ID
            FROM physionet-data.mimiciii_clinical.icustays ic
            WHERE ic.DBSOURCE = 'metavision'
        )        
        """


def _add_decimal_to_icd9_codes(icd9_codes: pd.Series) -> pd.Series:
    """Add a decimal point after the third digit of ICD9 codes.

    Args:
        icd9_codes (pd.Series): Column containing ICD9 codes.

    Returns
        pd.Series: Column containing ICD9 codes with decimals.
    """

    # add decimal point between 3rd and 4th digit
    icd9_codes = icd9_codes.str[:3] + "." + icd9_codes.str[3:]

    icd9_codes = pd.to_numeric(icd9_codes, errors="coerce")

    return icd9_codes


@data_loaders.register("all_diagnoses")
def load_diagnoses(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load diagnoses dataframe with all diagnoses for each admission. Merge
    admission_disharge_timestamps with diagnoses to get the timestamp for each
    diagnosis using admission_ids.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Diagnoses table.
    """

    if nrows:
        base_query += f"LIMIT {nrows}"

    diagnoses = load_sql_query(base_query)

    # remove all rows with NaN values in ICD9_CODE
    diagnoses = diagnoses.dropna(subset=["ICD9_CODE"])

    diagnoses = diagnoses.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "HADM_ID": "admission_id",
        },
    )

    admission_discharge_timestamps = load_admission_timestamps()

    # Merge admission timestamps
    diagnoses = pd.merge(diagnoses, admission_discharge_timestamps, on="admission_id")

    # Rename admission timestamp column
    diagnoses = diagnoses.rename(columns={"admission_timestamp": "timestamp"})

    return diagnoses.reset_index(drop=True)


@data_loaders.register("a_diagnoses")
def load_a_diagnoses(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all A-diagnoses for each admission."""

    diagnoses = load_diagnoses(nrows=nrows)

    return diagnoses[diagnoses["SEQ_NUM"] == 1]


def _load_icd9_range(
    icd9_range: tuple[int, int],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all diagnoses for a given range of ICD9 codes.

    Args:
        icd9_range (tuple[int,int]): Tuple of start and end of ICD9 codes range.
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Diagnoses table.
    """

    df = load_diagnoses(nrows=nrows)

    # discard diagnoses that start with V or E
    df = df[~df["ICD9_CODE"].str.startswith(("V", "E"))]

    df["ICD9_CODE"] = _add_decimal_to_icd9_codes(df["ICD9_CODE"])

    # keep only rows with icd9 codes in range of interest
    df = df[df["ICD9_CODE"].between(icd9_range[0], icd9_range[1])]

    # add value col for feature generation
    df["value"] = 1

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("metastatic_cancer")
def load_metastatic_cancer(
    icd9_range: tuple[int, int] = (196, 199.9),
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all metastatic cancer A-diagnoses for each admission."""

    df = _load_icd9_range(icd9_range, nrows)

    return df.reset_index(drop=True)


@data_loaders.register("hematologic_malignancy")
def load_hematologic_malignancy(
    icd9_range: tuple[int, int] = (200, 209.9),
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all hematologic malignancy A-diagnoses for each admission."""

    df = _load_icd9_range(icd9_range, nrows)

    return df.reset_index(drop=True)


@data_loaders.register("acquired_immunodeficiency_syndrome")
def load_acquired_immunodeficiency_syndrome(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all acquired immunodeficiency syndrome A-diagnoses for each
    admission."""

    df = load_diagnoses(nrows=nrows)

    # keep only rows where ICD9_CODE is 042, V08, or 079.53
    df = df[df["ICD9_CODE"].isin(["042", "V08", "07953"])]

    # add value col for feature generation
    df["value"] = 1

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


if __name__ == "__main__":
    specific_diag = load_acquired_immunodeficiency_syndrome(nrows=100000)
    diagnoses = load_diagnoses()
