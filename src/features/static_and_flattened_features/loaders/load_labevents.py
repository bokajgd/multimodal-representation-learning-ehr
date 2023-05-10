"""Loaders for the labevents table.

Contains all charted data for all patients.
"""

import time
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import (
    DATA_PATH,
    load_dataset_from_file,
    _drop_rows_with_too_small_value_frequency,
    )


@data_loaders.register("labevents")
def load_labevents(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the labevents table.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Labevents table.
    """

    file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "LABEVENTS.csv.gz"
    )


    df = load_dataset_from_file(
        file_path=file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "ITEMID",
            "CHARTTIME",
            "VALUENUM",
            "VALUEUOM",
        ],
    )

    # Remove rows where VALUENUM is NaN or 0
    df = df[df["VALUENUM"].notna()]
    df = df[df["VALUENUM"] != 0]

    # Rename columns
    df = df.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "CHARTTIME": "timestamp",
            "VALUENUM": "value",
        },
    )

    # Convert to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"]
                                     )
    return df.reset_index(drop=True)    


@data_loaders.register("urea_nitrogen")
def load_urea_nitrogen(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    
    labevents = load_labevents(nrows=nrows)

    df = labevents[labevents["ITEMID"] == 51006]

    # Drop rows with measuring unit (VALUEUOM) that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df,
        value_col_name="VALUEUOM",
        threshold=0.1,
    )

    return df[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


@data_loaders.register("bicarbonate")
def load_bicarbonate(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    
    labevents = load_labevents(nrows=nrows)

    df = labevents[labevents["ITEMID"] == 50882]

    # Drop rows with measuring unit (VALUEUOM) that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df,
        value_col_name="VALUEUOM",
        threshold=0.1,
    )

    return df[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


@data_loaders.register("white_blod_cells")
def load_white_blod_cells(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    
    labevents = load_labevents(nrows=nrows)

    df = labevents[labevents["ITEMID"].isin([51300, 51301])]

    # Drop rows with measuring unit (VALUEUOM) that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df,
        value_col_name="VALUEUOM",
        threshold=0.1,
    )

    return df[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


@data_loaders.register("sodium_level")
def load_sodium_level(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    
    labevents = load_labevents(nrows=nrows)

    df = labevents[labevents["ITEMID"].isin([950824, 50983])]

    # Drop rows with measuring unit (VALUEUOM) that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df,
        value_col_name="VALUEUOM",
        threshold=0.1,
    )

    return df[['patient_id', 'timestamp', 'value']].reset_index(drop=True)



@data_loaders.register("potassium_level")
def load_potassium_level(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    
    labevents = load_labevents(nrows=nrows)

    df = labevents[labevents["ITEMID"].isin([50822, 50971])]

    # Drop rows with measuring unit (VALUEUOM) that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df,
        value_col_name="VALUEUOM",
        threshold=0.1,
    )

    return df[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


@data_loaders.register("bilirubin_level")
def load_bilirubin_level(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    
    labevents = load_labevents(nrows=nrows)

    df = labevents[labevents["ITEMID"] == 50885]

    # Drop rows with measuring unit (VALUEUOM) that only appears in less than 10% of the rows
    df = _drop_rows_with_too_small_value_frequency(
        df=df,
        value_col_name="VALUEUOM",
        threshold=0.1,
    )

    return df[['patient_id', 'timestamp', 'value']].reset_index(drop=True)


if __name__ == "__main__":
    start_time = time.time()
    df = load_bilirubin_level(nrows=100000)
    print(f"Time to load: {time.time() - start_time:.2f} seconds")