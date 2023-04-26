"""Loaders for the free-text notes.
"""
import time
from typing import Any, Optional

import pandas as pd

from timeseriesflattener.utils import data_loaders
from utils import load_sql_query

BASE_QUERY = """
        SELECT *
        FROM physionet-data.mimiciii_notes.noteevents ne
        WHERE ne.HADM_ID IN (SELECT DISTINCT HADM_ID FROM physionet-data.mimiciii_clinical.inputevents_mv)
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

    return df

if __name__ == "__main__":
    start_time = time.time()
    notes = load_notes(nrows=1000)
    print(f"Time to load notess: {time.time() - start_time:.2f} seconds")