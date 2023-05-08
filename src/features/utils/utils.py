"""Misc.

functions for feature generation.
"""

import numpy as np
import pandas as pd
from static_and_flattened_features.loaders.load_demographics import load_dob


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add age to a feature dataframe.

    Args:
        df (pd.DataFrame): Dataframe to add age to.

    Returns:
        pd.DataFrame: Dataframe with age added.
    """

    birthdays = load_dob()
    birthdays["date_of_birth"] = pd.to_datetime(birthdays["date_of_birth"])

    df = df.merge(
        birthdays,
        on="patient_id",
        how="left",
    )

    # Use .apply to combat OverflowError due to dates being offset
    df["pred_age"] = df.apply(
        lambda x: round(
            (x["timestamp"].to_pydatetime() - x["date_of_birth"].to_pydatetime()).days
            / 365,
            2,
        ),
        axis=1,
    )

    df = df.drop(columns=["date_of_birth"])

    return df.reset_index(drop=True)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from the dataset. All values in numeric columns that are
    in the top or bottom 1% of the distribution are set to 0.

    Args:
        df (pd.DataFrame): Dataframe to remove outliers from.

    Returns:
        pd.DataFrame: Dataframe with outliers removed.
    """

    # select only columns with numeric data types and drop patient_id column
    numeric_cols = df.select_dtypes(include=np.number).drop(
        columns=["patient_id"],
    )

    # disregard columns with only two unique values to avoid binarizing columns that are already binary
    numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 2]

    # disregard text features columns
    non_text_feature_columns = [
        column for column in numeric_cols.columns if not column.startswith("pred_text_")
    ]
    numeric_cols = numeric_cols[non_text_feature_columns]

    # iterate over each column and set outliers to 0
    for column in numeric_cols.columns:

        # disregard 0 values when calculating bottom quantile
        bottom_quantile = (
            numeric_cols[column].loc[numeric_cols[column] != 0].quantile(0.01)
        )
        top_quantile = numeric_cols[column].quantile(0.99)

        is_outlier = (numeric_cols[column] > top_quantile) | (
            numeric_cols[column] < bottom_quantile
        )
        numeric_cols.loc[is_outlier, column] = 0

    # replace original numeric columns with outlier-free columns
    df[numeric_cols.columns] = numeric_cols

    return df.reset_index(drop=True)
