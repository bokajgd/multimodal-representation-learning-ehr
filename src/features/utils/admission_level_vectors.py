"""Copy code."""
import numpy as np
import pandas as pd


def aggregate_co_vectors(
    co_df: pd.DataFrame,
    binary_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Function for aggregating co-occurrence vectors for each admission to
    create admission-level patient vectors.

    Args:
        co_df: pandas DataFrame containing the co-occurrence vectors for each admission.
        binary_feature_df: pandas DataFrame containing the binary features for each admission.

    Returns:
        aggregated_feature_vectors: pandas DataFrame containing the aggregated feature vectors for each admission.
    """

    feature_cols_df = binary_feature_df.filter(regex=r"^pred_")
    other_cols_df = binary_feature_df.drop(columns=feature_cols_df.columns)

    # convert the co-occurrence matrix to a numpy array
    co_array = co_df.to_numpy()

    # initialize an empty list to store the aggregated feature vectors
    aggregated_feature_vectors = []

    # iterate over each row in the feature dataframe
    for _, row in feature_cols_df.iterrows():
        # extract binary feature vector for that row
        feature_vector = row.values

        # find the indices of the features that are 1 in the current row
        feature_indices = np.where(feature_vector == 1)[0]

        # extract the co-occurrence vectors for the selected features
        selected_co_vectors = co_array[feature_indices]

        # calculate the aggregated feature vector by taking the mean
        aggregated_vector = np.mean(selected_co_vectors, axis=0)

        # append the aggregated feature vector to the result list
        aggregated_feature_vectors.append(aggregated_vector)

    # convert the aggregated feature vectors list to a pandas DataFrame
    aggregated_dataframe = pd.DataFrame(
        aggregated_feature_vectors,
        columns=co_df.columns,
    )

    # round all values to 4 decimal places
    aggregated_dataframe = aggregated_dataframe.round(4)

    # concatenate the aggregated feature vectors with the other columns
    aggregated_dataframe = pd.concat([other_cols_df, aggregated_dataframe], axis=1)

    return aggregated_dataframe
