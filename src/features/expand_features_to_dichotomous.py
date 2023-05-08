"""Script for expanding numerical features to binary features."""

import numpy as np
import pandas as pd


def expand_numeric_cols_to_binary_percentile_cols(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand numerical columns to binary columns based on percentile ranges.
    From each column with numerical data, it creates 3 new columns with binary
    values representing whether the original value is among the top 15%
    percentile, in the 15-85% range, or the bottom 15% percentile of the values
    in the column. Disregards nan values (patients missing data). For all the
    text columns, it converts all tfidf values that are not 0 to 1.

    Args
        feature_df (pd.DataFrame): DataFrame with flattened features.

    Returns:
        pd.DataFrame: DataFrame with expanded features.
    """

    # subset all text columns
    text_cols = feature_df.filter(regex=r"^pred_text_")

    # set all values in the text_cols that are not 0 to 1
    text_cols = text_cols.applymap(lambda x: 1 if x != 0 else 0)

    # select only columns with numeric data types and drop patient_id column and outcome column and text_cols
    numeric_cols = (
        feature_df.select_dtypes(include=["number"])
        .drop(
            columns=[
                "patient_id",
                "outc_date_of_death_within_30_days_bool_fallback_0_dichotomous",
            ],
        )
        .drop(columns=text_cols.columns)
    )

    # disregard columns with only two unique values to avoid binarizing columns that are already binary
    numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 2]

    # initialize an empty list to store the new DataFrames for each column
    expanded_data = []

    # iterate over each column
    for col in numeric_cols.columns:

        col_data = numeric_cols[col].values

        # calc the percentiles for the column data
        p15, p85 = np.nanpercentile(col_data, [15, 85])

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
            feature_df[
                [
                    "patient_id",
                    "timestamp",
                    "prediction_time_uuid",
                    "outc_date_of_death_within_30_days_bool_fallback_0_dichotomous",
                ]
            ],
            *expanded_data,
        ],
        axis=1,
    )

    # concatenate the text columns with the expanded columns
    output_data = pd.concat([output_data, text_cols], axis=1)

    # return the binarised feature df
    return output_data
