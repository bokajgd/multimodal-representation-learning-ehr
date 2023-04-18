"""Create full dataset with expanded features and save to disk."""
from generate_flattened_features import generate_flattened_features
from expand_features_to_dichotomous import expand_numeric_cols_to_binary_percentile_cols

from utils.project_setup import (
    get_project_info,
)
from utils.cooccurence_counts import (
    calculate_co_occurrence,
)

def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    project_info = get_project_info()

    flattened_df = generate_flattened_features(save_to_disk=False)

    expanded_df = expand_numeric_cols_to_binary_percentile_cols(
        feature_df=flattened_df,
    )
    
    co_df = calculate_co_occurrence(df=expanded_df)

    if project_info.dataset_format == "parquet":
        expanded_df.to_parquet(project_info.feature_set_path / "expanded_features.parquet")
    elif project_info.dataset_format == "csv":
        expanded_df.to_csv(project_info.feature_set_path / "expanded_features.csv")

if __name__ == "__main__":
    main()