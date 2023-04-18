"""Script for expanding numerical features to binary features."""

import numpy as np
import pandas as pd


def expand_numeric_cols_to_binary_percentile_cols(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand numerical columns to binary columns based on percentile ranges.
    From each column with numerical data, it creates 7 new columns with binary
    values representing whether the original value is among the top 1%, 1-5%,
    5-25%, 25-75%, 75-95%, 95%-99%, 99-100% of the values in the column.

    Args:
        feature_df (pd.DataFrame): DataFrame with flattened features.

    Returns:
        pd.DataFrame: DataFrame with expanded features.
    """

    # select only columns with numeric data types and drop patient_id column
    numeric_cols = feature_df.select_dtypes(include=np.number).drop(
        columns=["patient_id"],
    )

    # remove columns with only two unique values to avoid binarizing columns that are already binary
    numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 2]

    # initialize an empty list to store the new DataFrames for each column
    expanded_data = []

    # iterate over each column
    for col in numeric_cols.columns:

        col_data = numeric_cols[col].values

        # calc the percentiles for the column data
        p15, p85 = np.percentile(col_data, [15, 85])

        # calc new binary columns for each percentile range
        col_p15 = np.where(col_data <= p15, 1, 0)
        col_p_mid = np.where((col_data > p15) & (col_data <= p85), 1, 0)
        col_p85 = np.where(col_data > p85, 1, 0)

        # create a new df with the expanded columns for the current column
        new_data = pd.DataFrame(
            {f"{col}_p15": col_p15, f"{col}_p_mid": col_p_mid, f"{col}_p85": col_p85},
        )

        # append the new DataFrame to the list of expanded DataFrames
        expanded_data.append(new_data)

    # concatenate all the expanded columns with the patient_id, timestamp and prediction_time_uuid columns
    output_data = pd.concat(
        [
            feature_df[["patient_id", "timestamp", "prediction_time_uuid"]],
            *expanded_data,
        ],
        axis=1,
    )

    # return the binarised feature df
    return output_data
