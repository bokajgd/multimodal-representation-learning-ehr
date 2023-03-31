"""Loaders for the inputevents table.

Inputevents covers all fluids that are administered to patients via IV
or other routes. Inputevents loaders data register naming convention:
'<ITEMID>_<AMOUNTUOM>'
"""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import (
    DATA_PATH,
    _drop_rows_with_too_small_patient_or_admission_frequency,
    _drop_rows_with_too_small_value_frequency,
    load_dataset_from_file,
)


@data_loaders.register("inputevents")
def load_inputevents(
    nrows: Optional[int] = None,
    load_for_flattening: bool = True,
) -> pd.DataFrame:
    """Load inputevents table.

    Args:
        nrows (int): Number of rows to load.
        load_for_flattening (bool): If True, loads the table for feature generation and only keeps 'patient_id',
          'timestamp' and 'value' colums. If False, keeps all columns for further processing Defaults to True.

    Returns:
        pd.DataFrame: inputevents table.
    """
    inputevents_file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "INPUTEVENTS_MV.csv.gz"
    )

    inputevents = load_dataset_from_file(
        file_path=inputevents_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "ITEMID",
            "STARTTIME",
            "AMOUNT",
            "AMOUNTUOM",
            "PATIENTWEIGHT",
            "CANCELREASON",
        ],
    )

    # Remove cancelled events
    inputevents = inputevents[inputevents["CANCELREASON"] == 0].drop(
        columns=["CANCELREASON"],
    )

    # Remove all items that have only been administered to less than 10% of the patients
    inputevents = _drop_rows_with_too_small_patient_or_admission_frequency(
        df=inputevents,
        patient_or_admission_col_name="SUBJECT_ID",
        item_col_name="ITEMID",
        threshold=0.05,
    )

    # Rename columns
    inputevents = inputevents.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "STARTTIME": "timestamp",
        },
    )

    # Convert to datetime
    inputevents["timestamp"] = pd.to_datetime(inputevents["timestamp"])

    if load_for_flattening:

        inputevents["value"] = 1

        # Keep only columns for feature generation
        return inputevents[["patient_id", "timestamp", "value"]].reset_index(drop=True)

    else:

        return inputevents.reset_index(drop=True)


@data_loaders.register("weight")
def load_weight(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    # Keep only columns with patient_id, weight and timestamp and drop duplicates
    df = inputevents[["patient_id", "timestamp", "PATIENTWEIGHT"]].drop_duplicates()

    df = df.rename(columns={"PATIENTWEIGHT": "value"})

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


def _calc_amount_to_bodyweight_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the amount to bodyweight ratio for each row in the
    inputevents table."""

    return df["AMOUNT"] / df["PATIENTWEIGHT"]


@data_loaders.register("fentanyl_mg")
def load_fentanyl(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    # Filter for fentanyl (221744, 225972 and 225942)
    df = inputevents[inputevents["ITEMID"].isin([221744, 225942])]

    # Drop rows with amountuom value that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df, value_col_name="AMOUNTUOM", threshold=0.1
    )

    # Convert mcg to mg
    df.loc[df["AMOUNTUOM"] == "mcg", "AMOUNT"] = df["AMOUNT"] / 1000

    # Calculate fentanyl amount/bodyweight ratio
    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    # Keep only columns for feature generation
    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("albumin_5_ml")
def load_albumin_5(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    df = inputevents[inputevents["ITEMID"].isin([220864])]

    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("fresh_frozen_plasma_ml")
def load_fresh_frozen_plasma(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    df = inputevents[inputevents["ITEMID"].isin([220970])]

    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("sterile_water_ml")
def load_sterile_water(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    df = inputevents[inputevents["ITEMID"].isin([225944])]

    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("potassium_chloride_meq")
def load_potassium_chloride(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    df = inputevents[inputevents["ITEMID"].isin([225166])]

    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("gastric_meds_ml")
def load_gastric_meds(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    df = inputevents[inputevents["ITEMID"].isin([225799])]

    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


@data_loaders.register("nacl_0_9_ml")
def load_nacl_0_9(
    nrows: Optional[int] = None,
) -> pd.DataFrame:

    inputevents = load_inputevents(nrows=nrows, load_for_flattening=False)

    df = inputevents[inputevents["ITEMID"].isin([225158])]

    df = _drop_rows_with_too_small_value_frequency(
        df=df, value_col_name="AMOUNTUOM", threshold=0.1
    )

    df["value"] = _calc_amount_to_bodyweight_ratio(df)

    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


if __name__ == "__main__":
    df = load_weight()
