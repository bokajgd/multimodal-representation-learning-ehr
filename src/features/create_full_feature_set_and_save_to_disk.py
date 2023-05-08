"""Create full dataset with expanded features and save to disk."""
import logging
import time

from expand_features_to_dichotomous import expand_numeric_cols_to_binary_percentile_cols
from generate_flattened_features import generate_flattened_features
from utils.admission_level_vectors import aggregate_co_vectors
from utils.cooccurence_counts import calculate_co_occurrence
from utils.project_setup import get_project_info


def main(save_to_disk: bool = False, min_set_for_debug: bool = False):
    """Main function for loading, generating and evaluating a flattened
    dataset."""

    logging.info("Loading project info...")
    project_info = get_project_info()

    logging.info("Generating features...")
    flattened_df = generate_flattened_features(
        save_to_disk=save_to_disk,
        min_set_for_debug=min_set_for_debug,
    )

    logging.info("Binary-encoding features...")
    binary_feature_df = expand_numeric_cols_to_binary_percentile_cols(
        feature_df=flattened_df,
    )

    logging.info("Calculating co-occurrence counts...")
    co_df = calculate_co_occurrence(df=binary_feature_df)

    logging.info("Aggregating co-occurrence counts to admission level vectors...")
    admission_level_vectors_df = aggregate_co_vectors(
        co_df=co_df,
        binary_feature_df=binary_feature_df,
    )

    if save_to_disk:
        if project_info.dataset_format == "parquet":
            binary_feature_df.to_parquet(
                project_info.feature_set_path / "binary_feature_df.parquet",
            )
            co_df.to_parquet(
                project_info.feature_set_path / "co_df.parquet",
            )
            admission_level_vectors_df.to_parquet(
                project_info.feature_set_path / "admission_level_vectors_df.parquet",
            )
        elif project_info.dataset_format == "csv":
            binary_feature_df.to_csv(
                project_info.feature_set_path / "binary_feature_df.csv",
            )
            co_df.to_csv(project_info.feature_set_path / "co_counts.csv")
            admission_level_vectors_df.to_csv(
                project_info.feature_set_path / "admission_level_vectors_df.csv",
            )

    binary_feature_df.to_csv(project_info.feature_set_path / "binary_feature_df.csv")
    co_df.to_csv(project_info.feature_set_path / "co_counts.csv")
    admission_level_vectors_df.to_csv(
        project_info.feature_set_path / "admission_level_vectors_df.csv",
    )

    return None


if __name__ == "__main__":
    main()
