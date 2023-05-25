"""Loaders for the chartevents table.

Contains all charted data for all patients.
"""

import time
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import DATA_PATH, load_dataset_from_file, load_sql_query

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


@data_loaders.register("pao2_fio2_ratio")
def load_pao2_fio2_ratio(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load pao2/fio2 ratio."""

    # Load fio2 data
    QUERY = base_query + "AND ce.ITEMID IN (223835)"

    if nrows:
        QUERY += f"LIMIT {nrows}"

    fio2_df = load_sql_query(QUERY)

    # if value is between 0 and 1, multiply by 100 to get percentage
    fio2_df.loc[
        (fio2_df["VALUENUM"] > 0) & (fio2_df["VALUENUM"] < 1),
        "VALUENUM",
    ] *= 100

    # if value is between 1 and 21, remove from dataset as they are likely errors
    fio2_df = fio2_df[~((fio2_df["VALUENUM"] > 1) & (fio2_df["VALUENUM"] < 21))]

    # laod pao2 data
    file_path = DATA_PATH / "mimic-iii-clinical-database-1.4" / "LABEVENTS.csv.gz"

    pao2_df = load_dataset_from_file(
        file_path=file_path,
        cols_to_load=[
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID",
            "CHARTTIME",
            "VALUENUM",
        ],
    )

    pao2_df = pao2_df[pao2_df["ITEMID"] == 50821]

    # drop ITEMID column
    pao2_df.drop(columns=["ITEMID"], inplace=True)

    # drop rows with nan values in the VALLUENUM column or HADM_ID column
    pao2_df = pao2_df.dropna(subset=["VALUENUM", "HADM_ID"])
    fio2_df = fio2_df.dropna(subset=["VALUENUM", "HADM_ID"])

    # drop rows with 0 in VALUENUM columns
    pao2_df = pao2_df[pao2_df["VALUENUM"] != 0]
    fio2_df = fio2_df[fio2_df["VALUENUM"] != 0]

    # convert admission IDs to integers
    pao2_df["HADM_ID"] = pao2_df["HADM_ID"].astype(int)
    fio2_df["HADM_ID"] = fio2_df["HADM_ID"].astype(int)

    # convert the CHARTTIME columns in both DataFrames to datetime objects
    pao2_df["CHARTTIME"] = pd.to_datetime(pao2_df["CHARTTIME"])
    fio2_df["CHARTTIME"] = pd.to_datetime(fio2_df["CHARTTIME"])

    # keep only rows with HADM_IDs that are in both DataFrames
    pao2_df = pao2_df[pao2_df["HADM_ID"].isin(fio2_df["HADM_ID"])]
    fio2_df = fio2_df[fio2_df["HADM_ID"].isin(pao2_df["HADM_ID"])]

    # Group pao2 and fio2 DataFrames by HADM_ID
    pao2_grouped = pao2_df.groupby("HADM_ID")
    fio2_grouped = fio2_df.groupby("HADM_ID")

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(
        columns=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "PaO2", "FiO2", "ratio"],
    )

    # Iterate over each HADM_ID
    for hadm_id in pao2_df["HADM_ID"].unique():
        # get the pao2 measurements for the current HADM_ID
        hadm_pao2 = pao2_grouped.get_group(hadm_id)

        # get the fio2 measurements for the current HADM_ID
        hadm_fio2 = fio2_grouped.get_group(hadm_id)

        # Iterate over each pao2 measurement
        for index, row in hadm_pao2.iterrows():
            # Get the most recent prior fio2 measurement for the current pao2 measurement
            recent_fio2 = hadm_fio2[
                (hadm_fio2["CHARTTIME"] < row["CHARTTIME"])
                & (hadm_fio2["CHARTTIME"] >= row["CHARTTIME"] - pd.Timedelta(hours=2))
            ].head(1)

            # Calculate the pao2/fio2 ratio and append the result to the result DataFrame
            if not recent_fio2.empty:
                ratio = row["VALUENUM"] / recent_fio2["VALUENUM"].values[0]
                subset_df = pd.DataFrame(
                    {
                        "SUBJECT_ID": row["SUBJECT_ID"],
                        "HADM_ID": hadm_id,
                        "CHARTTIME": row["CHARTTIME"],
                        "PaO2": row["VALUENUM"],
                        "FiO2": recent_fio2["VALUENUM"].values[0],
                        "ratio": ratio * 100,
                    },
                    index=[0],
                )

                result_df = pd.concat([result_df, subset_df], ignore_index=True)

    # keep only the relevant columns and rename to match the format of the other tables
    result_df = result_df[["SUBJECT_ID", "CHARTTIME", "ratio"]].rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "CHARTTIME": "timestamp",
            "ratio": "value",
        },
    )

    # Convert timestamp to datetime
    result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])

    return result_df.reset_index(drop=True)


if __name__ == "__main__":
    # df = load_pao2_fio2_ratio()
    start_time = time.time()
    df = load_pao2_fio2_ratio()
    print(f"Time to load: {time.time() - start_time:.2f} seconds")
