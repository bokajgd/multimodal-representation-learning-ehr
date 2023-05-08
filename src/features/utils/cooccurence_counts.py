import numpy as np
import pandas as pd


def calculate_co_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the co-occurrence counts for all the binary features (all the
    columns that start with 'pred_').

    Args:
        df: pandas DataFrame containing the binary features.

    Returns:
        co_df: pandas DataFrame containing co-occurrence counts.
    """

    # Get the columns that start with 'pred_'
    pred_cols = [col for col in df.columns if col.startswith("pred_")]

    # Create an empty co-occurrence matrix
    co_occurrence_matrix = np.zeros((len(pred_cols), len(pred_cols)), dtype=np.int32)

    # Loop through each row in the DataFrame
    for _, row in df.iterrows():
        # Get the binary features for the row
        binary_features = row[pred_cols].values.astype(bool)

        # Compute the indices of binary features that are True
        true_indices = np.where(binary_features)[0]

        # Increment the co-occurrence count for all pairs of true indices
        for i in range(len(true_indices)):
            for j in range(i + 1, len(true_indices)):
                feature_i = true_indices[i]
                feature_j = true_indices[j]
                co_occurrence_matrix[feature_i, feature_j] += 1
                co_occurrence_matrix[feature_j, feature_i] += 1

    # Convert the NumPy array into a pandas DataFrame
    co_occurrence_df = pd.DataFrame(
        co_occurrence_matrix,
        index=pred_cols,
        columns=pred_cols,
    )

    # normalize the co-occurrence counts
    normalized_co_occurrence_df = (co_occurrence_df - co_occurrence_df.min().min()) / (
        co_occurrence_df.max().max() - co_occurrence_df.min().min()
    )

    return normalized_co_occurrence_df
