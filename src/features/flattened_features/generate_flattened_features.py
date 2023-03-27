"""Main feature generation."""
from feautre_specification.specify_features import FeatureSpecifier
from loaders.load_admissions import load_emergency_admissions
from loaders.utils import DATA_PATH

from utils.flatten_dataset import (
    create_flattened_dataset,
)
from utils.project_setup import (
    get_project_info,
)

import pandas as pd


def generate_flattened_features(save_to_disk: bool = False) -> pd.DataFrame:
    """Main function for loading, generating and evaluating a flattened
    dataset."""

    project_info = get_project_info()

    feature_specs = FeatureSpecifier(
        project_info=project_info,
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=load_emergency_admissions(timestamps_only=True),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )

    if save_to_disk:
        if project_info.dataset_format == "parquet":
            flattened_df.to_parquet(project_info.feature_set_path / "flattened_features.parquet")
        elif project_info.dataset_format == "csv":
            flattened_df.to_csv(project_info.feature_set_path / "flattened_features.csv")

    return flattened_df

if __name__ == "__main__":
    generate_flattened_features(save_to_disk=True)