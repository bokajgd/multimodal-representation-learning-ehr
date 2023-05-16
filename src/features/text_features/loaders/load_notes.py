"""Loaders for the free-text notes."""

import time
from typing import Any, Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import text_preprocessing, DATA_PATH, load_dataset_from_file


@data_loaders.register("noteevents")
def load_notes(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all notes.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Table with preprocessed notes.
    """
    noteevents_file_path = (
        DATA_PATH / "mimic-iii-clinical-database-1.4" / "NOTEEVENTS.csv.gz"
    )

    df = load_dataset_from_file(
        file_path=noteevents_file_path,
        nrows=nrows,
        cols_to_load=[
            "SUBJECT_ID",
            "CHARTTIME",
            "TEXT",
            "ISERROR",
            "HADM_ID",
        ],
    )

    # Rename columns
    df = df.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "CHARTTIME": "timestamp",
            "TEXT": "text",
        }
    )

    predictions_times_df_path = DATA_PATH / "misc" / "cohort_with_prediction_times.csv"
    prediction_times_df = pd.read_csv(predictions_times_df_path).drop(
        columns="timestamp"
    )

    # Only keep notes from admissions that are in the prediction_times_df
    df = df.merge(prediction_times_df, on=["patient_id", "HADM_ID"])

    # Remove rows where error is not null
    df = df[df["ISERROR"].isna()].drop(columns=["ISERROR", "HADM_ID"])

    # Remove rows where CHARTTIME is NaN
    df = df.dropna(subset=["timestamp"])

    # Preprocess text
    df = text_preprocessing(df)

    return df


if __name__ == "__main__":
    start_time = time.time()
    notes = load_notes(nrows=800000)
    print(f"Time to load notes: {time.time() - start_time:.2f} seconds")
    print(notes.head())
