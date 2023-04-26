"""Loaders for the chartevents table.

Contains all charted data for all patients.
"""

import time
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from utils import load_sql_query

BASE_QUERY = """
        SELECT ce.SUBJECT_ID, ce.HADM_ID, ce.ITEMID, ce.VALUE, ce.VALUENUM, ce.VALUEUOM, ce.CHARTTIME,
        FROM physionet-data.mimiciii_clinical.chartevents ce
        WHERE ce.HADM_ID IN (SELECT DISTINCT HADM_ID FROM physionet-data.mimiciii_clinical.inputevents_mv)
        """

@data_loaders.register("noteevents")
def load_chartevents(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the chartevents table.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Chartevents table.
    """

    if nrows:
        base_query += f"LIMIT {nrows}"

    df = load_sql_query(base_query)

    return df

@data_loaders.register("gcs")
def load_gcs(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = 10000,
) -> pd.DataFrame:
    """Load data for the Glasgow coma scale."""

    QUERY = base_query + "AND ce.ITEMID IN (723, 454, 184, 223900, 223901, 220739)"

    if nrows:
        QUERY += f"LIMIT {nrows}"

    df = load_sql_query(QUERY)

    # Drop rows with nan values in the VALLUENUM column
    df = df.dropna(subset=["VALUENUM"])

    # Keep only the relevant columns and rename to match the format of the other tables
    df = df[["SUBJECT_ID", "VALUENUM", "CHARTTIME"]].rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "VALUENUM": "value",
            "CHARTTIME": "timestamp",
        },
    )

    # Collapse all rows with identical patient_id and timestamp into one row with the summed value to calculate the GCS
    df = df.groupby(["patient_id", "timestamp"]).sum().reset_index()

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.reset_index(drop=True)


if __name__ == "__main__":

    start_time = time.time()
    gcs = load_gcs()
    print(f"Time to load gcs: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    df = load_chartevents()
    print(f"Time to load chartevents: {time.time() - start_time:.2f} seconds")
