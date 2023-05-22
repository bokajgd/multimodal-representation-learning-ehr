"""Loaders for the outputevents table.

Outputevents covers all fluids which have either been excreted by the
patient, such as urine output, or extracted from the patient, for
example through a drain.Outputevents loaders data register naming
convention: '<ITEMID>_<AMOUNTUOM>'
"""

import time
from typing import Optional

import pandas as pd
from timeseriesflattener.utils import data_loaders

from .utils import (
    DATA_PATH,
    load_dataset_from_file,
)


@data_loaders.register("outputevents")
def load_outputevents(
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
    file_path = DATA_PATH / "mimic-iii-clinical-database-1.4" / "outputevents.csv.gz"

    outputevents = load_dataset_from_file(
        file_path=file_path,
        nrows=nrows,
        cols_to_load=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUEUOM"],
    )

    # Rename columns
    outputevents = outputevents.rename(
        columns={
            "SUBJECT_ID": "patient_id",
            "CHARTTIME": "timestamp",
            "VALUE": "value",
        },
    )

    # Remove rows where VALUE is NaN or 0
    outputevents = outputevents[outputevents["value"].notna()]
    outputevents = outputevents[outputevents["value"] != 0]

    # Convert to datetime
    outputevents["timestamp"] = pd.to_datetime(outputevents["timestamp"])

    if load_for_flattening:
        outputevents["value"] = 1

        # Keep only columns for feature generation
        return outputevents[["patient_id", "timestamp", "value"]].reset_index(drop=True)

    else:
        return outputevents.reset_index(drop=True)


@data_loaders.register("urine")
def load_urine(
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    urine_list = [
        40055,
        43175,
        40069,
        40094,
        40715,
        40473,
        40085,
        40057,
        40056,
        40405,
        40428,
        40086,
        40096,
        40651,
        226559,
        226560,
        226561,
        226584,
        226563,
        226564,
        226565,
        226567,
        226557,
        226558,
        227488,
        227489,
    ]

    df = load_outputevents(nrows=nrows, load_for_flattening=False)

    df = df[df["ITEMID"].isin(urine_list)]
    return df[["patient_id", "timestamp", "value"]].reset_index(drop=True)


if __name__ == "__main__":
    start_time = time.time()
    df = load_urine()
    print(f"Time to load: {time.time() - start_time:.2f} seconds")
