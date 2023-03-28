"""Loaders for the icu stays table."""
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders
from utils import DATA_PATH, load_dataset_from_file


@data_loaders.register("icu_stays")
def load_icu_stays(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load icu stays table. Drops admissions without chartevents data and
    duplicate rows. Returns a df with columns for patient_id, timestamp, and
    value.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Icu stays table.
    """
    icu_stays_file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "ICUSTAYS.csv.gz"
    )

    icu_stays = load_dataset_from_file(
        file_path=icu_stays_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "OUTTIME",
            "LOS",
        ],
    )

    # Convert to datetime
    icu_stays["OUTTIME"] = pd.to_datetime(icu_stays["OUTTIME"])

    # Rename columns
    icu_stays = icu_stays.rename(
        columns={"SUBJECT_ID": "patient_id", "OUTTIME": "timestamp", "LOS": "value"}
    )

    return icu_stays.reset_index(drop=True)


if __name__ == "__main__":
    load_icu_stays(nrows=10000)
