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
        WHERE ce.HADM_ID IN (
            SELECT DISTINCT ic.HADM_ID
            FROM physionet-data.mimiciii_clinical.icustays ic
            WHERE ic.DBSOURCE = 'metavision'
        )
        """


@data_loaders.register("chartevents")
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

    return df.reset_index(drop=True)


@data_loaders.register("gcs")
def load_gcs(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
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


@data_loaders.register("systolic_blood_pressure")
def load_systolic_blood_pressure(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load data for systolic blood pressure."""

    QUERY = base_query + "AND ce.ITEMID IN (51, 442, 455, 6701, 220179, 220050)"

    if nrows:
        QUERY += f"LIMIT {nrows}"

    df = load_sql_query(QUERY)

    # Drop rows with nan values in the VALLUENUM column or with a value of 0
    df = df.dropna(subset=["VALUENUM"])
    df = df[df["VALUENUM"] != 0]

    # Keep only the relevant columns and rename to match the format of the other tables
    df = df[["SUBJECT_ID", "VALUENUM", "CHARTTIME"]].rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "VALUENUM": "value",
            "CHARTTIME": "timestamp",
        },
    )

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.reset_index(drop=True)


@data_loaders.register("heart_rate")
def load_heart_rate(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load data for heart rate measurements."""

    QUERY = base_query + "AND ce.ITEMID IN (211, 220045)"

    if nrows:
        QUERY += f"LIMIT {nrows}"

    df = load_sql_query(QUERY)

    # Drop rows with nan values in the VALLUENUM column or with a value of 0
    df = df.dropna(subset=["VALUENUM"])
    df = df[df["VALUENUM"] != 0]

    # Keep only the relevant columns and rename to match the format of the other tables
    df = df[["SUBJECT_ID", "VALUENUM", "CHARTTIME"]].rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "VALUENUM": "value",
            "CHARTTIME": "timestamp",
        },
    )

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.reset_index(drop=True)


@data_loaders.register("temperature")
def load_temperature(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load data for body temperature measurements."""

    QUERY = base_query + "AND ce.ITEMID IN (223761, 678, 223762, 676)"

    if nrows:
        QUERY += f"LIMIT {nrows}"

    df = load_sql_query(QUERY)

    # Drop rows with nan values in the VALLUENUM column or with a value of 0
    df = df.dropna(subset=["VALUENUM"])
    df = df[df["VALUENUM"] != 0]

    # Convert temperature from Fahrenheit to Celsius (all values above 50 are assumed to be in Fahrenheit)
    df.loc[df["VALUENUM"] > 50, "VALUENUM"] = round(
        (df.loc[df["VALUENUM"] > 50, "VALUENUM"] - 32) * 5 / 9,
        2,
    )

    # Keep only the relevant columns and rename to match the format of the other tables
    df = df[["SUBJECT_ID", "VALUENUM", "CHARTTIME"]].rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "VALUENUM": "value",
            "CHARTTIME": "timestamp",
        },
    )

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df.reset_index(drop=True)


if __name__ == "__main__":

    start_time = time.time()
    df = load_temperature(nrows=10000)
    print(f"Time to load: {time.time() - start_time:.2f} seconds")
