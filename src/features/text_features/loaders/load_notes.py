"""Loaders for the free-text notes."""
import time
from typing import Any, Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import DATA_PATH, load_sql_query, text_preprocessing

BASE_QUERY = """
        SELECT ne.SUBJECT_ID AS patient_id, ne.CHARTTIME AS timestamp, ne.TEXT AS value, ne.ISERROR AS error
        FROM physionet-data.mimiciii_notes.noteevents ne
        WHERE ne.HADM_ID IN (SELECT DISTINCT HADM_ID FROM physionet-data.mimiciii_clinical.inputevents_mv)
        AND ne.CHARTTIME IS NOT NULL
        """


@data_loaders.register("noteevents")
def load_notes(
    base_query: str = BASE_QUERY,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load all notes.

    Args:
        nrows (int): Number of rows to load.

    Returns:
        pd.DataFrame: Chartevents table.
    """

    if nrows:
        base_query += f"LIMIT {nrows}"

    df = load_sql_query(base_query)

    # Remove rows where error is not null
    df = df[df["error"].isna()].drop(columns=["error"])

    # Preprocess text
    df = text_preprocessing(df)

    return df


if __name__ == "__main__":
    start_time = time.time()
    notes = load_notes(nrows=300)
    print(f"Time to load notes: {time.time() - start_time:.2f} seconds")
