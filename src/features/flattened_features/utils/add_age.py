"""Utility function for adding age to a feature dataframe."""

import pandas as pd

from loaders.load_demographics import load_dob


def add_age(df: pd.DataFrame,) -> pd.DataFrame:
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
    df['age'] = df.apply(lambda x: round((x['timestamp'].to_pydatetime() - x['date_of_birth'].to_pydatetime()).days/365, 2), axis=1)

    df = df.drop(columns=['date_of_birth'])

    return df.reset_index(drop=True)
