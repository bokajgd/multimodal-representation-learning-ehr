"""Create full dataset with expanded features and save to disk."""

from datetime import datetime

curr_timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

from typing import Optional

import pandas as pd
from expand_features_to_dichotomous import expand_numeric_cols_to_binary_percentile_cols
from generate_flattened_features import generate_flattened_features
from sklearn.model_selection import train_test_split
from utils.admission_level_vectors import aggregate_co_vectors
from utils.cooccurence_counts import calculate_co_occurrence
from utils.project_setup import get_project_info


def main(
    flattened_feature_set_path: Optional[str],
    read_flattened_features_from_disk: bool = False,
    save_to_disk: bool = True,
    min_set_for_debug: bool = False,
) -> None:
    """Main function for loading, generating and evaluating a flattened
    dataset."""

    print("Loading project info...", curr_timestamp)
    project_info = get_project_info()

    if read_flattened_features_from_disk:
        print("Reading flattened features from disk...")
        flattened_df = pd.read_csv(
            project_info.project_path
            / "data"
            / "feature_sets"
            / flattened_feature_set_path
            / "flattened_features.csv",
        )
    else:
        print("Generating features...", curr_timestamp)
        flattened_df = generate_flattened_features(
            save_to_disk=save_to_disk,
            min_set_for_debug=min_set_for_debug,
            saps_ii=True,
        )

    print("Binary-encoding features...", curr_timestamp)
    binary_feature_df = expand_numeric_cols_to_binary_percentile_cols(
        feature_df=flattened_df,
    )

    print("Calculating co-occurrence counts...", curr_timestamp)
    co_df = calculate_co_occurrence(df=binary_feature_df)

    print(
        "Aggregating co-occurrence counts to admission level vectors...", curr_timestamp
    )
    admission_level_vectors_df = aggregate_co_vectors(
        co_df=co_df,
        binary_feature_df=binary_feature_df,
    )

    print("Partioning data into train and test sets...", curr_timestamp)
    # define the train and test split sizes
    test_size = 0.3

    # split the data into train and test sets based on the patient ID and prediction time
    patients = flattened_df["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=42
    )

    train_flattened_df = flattened_df[flattened_df["patient_id"].isin(train_patients)]
    test_flattened_df = flattened_df[flattened_df["patient_id"].isin(test_patients)]

    # print the shapes of the train and test sets for the flattened_df
    print(f"Train set shape (flattened_df): {train_flattened_df.shape}")
    print(f"Test set shape (flattened_df): {test_flattened_df.shape}")

    # split the binary_feature_df into train and test sets based on the patient ID and prediction time
    train_binary_feature_df = binary_feature_df[
        binary_feature_df["patient_id"].isin(train_patients)
    ]
    test_binary_feature_df = binary_feature_df[
        binary_feature_df["patient_id"].isin(test_patients)
    ]

    # print the shapes of the train and test sets for the binary_feature_df
    print(f"Train set shape (binary_feature_df): {train_binary_feature_df.shape}")
    print(f"Test set shape (binary_feature_df): {test_binary_feature_df.shape}")

    # split the co_df into train and test sets based on the patient ID and prediction time
    train_admission_level_vectors_df = admission_level_vectors_df[
        admission_level_vectors_df["patient_id"].isin(train_patients)
    ]
    test_admission_level_vectors_df = admission_level_vectors_df[
        admission_level_vectors_df["patient_id"].isin(test_patients)
    ]

    # print the shapes of the train and test sets for the co_df
    print(
        f"Train set shape (admission_level_vectors_df): {train_admission_level_vectors_df.shape}"
    )
    print(
        f"Test set shape (admission_level_vectors_df): {test_admission_level_vectors_df.shape}"
    )

    if save_to_disk:
        test_flattened_df.to_csv(
            project_info.feature_set_path / "test_flattened_features.csv",
        )
        train_flattened_df.to_csv(
            project_info.feature_set_path / "train_flattened_features.csv",
        )
        train_binary_feature_df.to_csv(
            project_info.feature_set_path / "train_binary_feature_df.csv",
        )
        test_binary_feature_df.to_csv(
            project_info.feature_set_path / "test_binary_feature_df.csv",
        )
        train_admission_level_vectors_df.to_csv(
            project_info.feature_set_path / "train_admission_level_vectors_df.csv",
        )
        test_admission_level_vectors_df.to_csv(
            project_info.feature_set_path / "test_admission_level_vectors_df.csv",
        )
        co_df.to_csv(project_info.feature_set_path / "co_counts.csv")

    return None


if __name__ == "__main__":
    main(
        flattened_feature_set_path="multimodal_rep_learning_ehr_features_2023_05_09_15_09",
        read_flattened_features_from_disk=True,
    )
