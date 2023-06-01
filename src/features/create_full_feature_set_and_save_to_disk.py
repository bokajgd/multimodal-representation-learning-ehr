"""Full implementation of the pipeline"""

from datetime import datetime

curr_timestamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

from typing import Optional

import pandas as pd
from expand_features_to_dichotomous import expand_numeric_cols_to_binary_percentile_cols
from generate_flattened_features import generate_flattened_features
from sklearn.model_selection import train_test_split
from static_and_flattened_features.loaders.load_demographics import load_dod
from utils.admission_level_vectors import aggregate_co_vectors
from utils.cooccurence_counts import calculate_co_occurrence
from utils.project_setup import get_project_info


def main(
    flattened_feature_set_path: Optional[str],
    flattened_feature_set_filename: Optional[
        str
    ] = "full_flattened_features_with_text.csv",
    read_flattened_features_from_disk: bool = False,
    save_to_disk: bool = True,
    min_set_for_debug: bool = False,
    saps_ii: bool = True,
    get_text_features: bool = True,
) -> None:
    """Main function for generating a flattened feature dataset, binary features, co-occurence vectors, and patient embeddings."""

    print("Loading project info...", curr_timestamp)
    project_info = get_project_info()

    if read_flattened_features_from_disk:
        print("Reading flattened features from disk...")
        flattened_df = pd.read_csv(
            project_info.project_path
            / "data"
            / "feature_sets"
            / flattened_feature_set_path
            / flattened_feature_set_filename,
        )

        # if there is a column called "Unnamed: 0", drop it
        if "Unnamed: 0" in flattened_df.columns:
            flattened_df = flattened_df.drop("Unnamed: 0", axis=1)

        feature_set_prefix = flattened_feature_set_filename.split("_")[0]

    else:
        print("Generating features...", curr_timestamp)
        flattened_df, feature_set_prefix, project_info = generate_flattened_features(
            save_to_disk=save_to_disk,
            min_set_for_debug=min_set_for_debug,
            saps_ii=saps_ii,
            get_text_features=get_text_features,
        )

    flattened_df["sex_is_female"] = flattened_df["pred_sex_is_female"].copy()
    flattened_df["age"] = flattened_df["pred_age"].copy()

    print("Adding outcome timestamp col...", curr_timestamp)
    dod_df = load_dod()
    dod_df = dod_df.rename(columns={"timestamp": "date_of_death"})
    dod_df = dod_df.drop("value", axis=1)

    outc_col_name = [
        col_name for col_name in flattened_df.columns if col_name.startswith("outc_")
    ][0]

    flattened_df = pd.merge(flattened_df, dod_df, on="patient_id", how="left")
    flattened_df["outcome_timestamp"] = pd.NaT
    flattened_df.loc[
        flattened_df[outc_col_name] == 1,
        "outcome_timestamp",
    ] = flattened_df["date_of_death"]
    flattened_df = flattened_df.drop("date_of_death", axis=1)

    print("Binary-encoding features...", curr_timestamp)
    binary_feature_df = expand_numeric_cols_to_binary_percentile_cols(
        feature_df=flattened_df,
    )

    print("Calculating co-occurrence counts...", curr_timestamp)
    co_df = calculate_co_occurrence(df=binary_feature_df)

    print(
        "Aggregating co-occurrence counts to admission level vectors...",
        curr_timestamp,
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
        patients,
        test_size=test_size,
        random_state=42,
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
        f"Train set shape (admission_level_vectors_df): {train_admission_level_vectors_df.shape}",
    )
    print(
        f"Test set shape (admission_level_vectors_df): {test_admission_level_vectors_df.shape}",
    )

    if get_text_features:
        text_features_tag = "with_text_features"
    else:
        text_features_tag = "no_text_features"

    if save_to_disk:
        test_flattened_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{test_flattened_df.shape[0]}rows_{test_flattened_df.shape[1]}cols_test_flattened_features.csv",
            index=False,
        )
        train_flattened_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{train_flattened_df.shape[0]}rows_{train_flattened_df.shape[1]}cols_train_flattened_features.csv",
            index=False,
        )
        train_binary_feature_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{train_binary_feature_df.shape[0]}rows_{train_binary_feature_df.shape[1]}cols_train_binary_feature_df.csv",
            index=False,
        )
        test_binary_feature_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{test_binary_feature_df.shape[0]}rows_{test_binary_feature_df.shape[1]}cols_test_binary_feature_df.csv",
            index=False,
        )
        train_admission_level_vectors_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{train_admission_level_vectors_df.shape[0]}rows_{train_admission_level_vectors_df.shape[1]}cols_train_admission_level_vectors_df.csv",
            index=False,
        )
        test_admission_level_vectors_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{test_admission_level_vectors_df.shape[0]}rows_{test_admission_level_vectors_df.shape[1]}cols_test_admission_level_vectors_df.csv",
            index=False,
        )
        co_df.to_csv(
            project_info.feature_set_path
            / f"{feature_set_prefix}_{text_features_tag}_{co_df.shape[0]}rows_{co_df.shape[1]}cols_co_counts.csv",
            index=False,
        )

    return None


if __name__ == "__main__":
    main(
        flattened_feature_set_path="multimodal_rep_learning_ehr_features_2023_05_24_08_24",
        read_flattened_features_from_disk=False,
        save_to_disk=True,
        min_set_for_debug=False,
        saps_ii=False,
        get_text_features=False,
    )
