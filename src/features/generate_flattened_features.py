"""Main feature generation."""
import pandas as pd
from feature_specification.specify_features import FeatureSpecifier
from static_and_flattened_features.loaders.utils import DATA_PATH

from utils.add_age import add_age
from utils.flatten_dataset import create_flattened_dataset
from utils.project_setup import get_project_info


def generate_flattened_features(save_to_disk: bool = False) -> pd.DataFrame:
    """Main function for loading, generating and evaluating a flattened
    dataset."""

    predictions_times_df_path = DATA_PATH / "misc" / "cohort_with_prediction_times.csv"
    pred_times_df = pd.read_csv(predictions_times_df_path)

    # Keep only the last 1000 rows
    pred_times_df = pred_times_df.iloc[28500:].reset_index(drop=True)

    # Convert to datetime
    pred_times_df["timestamp"] = pd.to_datetime(
        pred_times_df["timestamp"],
    )

    project_info = get_project_info()

    feature_specs = FeatureSpecifier(
        project_info=project_info,
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=pred_times_df,
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )

    # Add age
    flattened_df = add_age(flattened_df)

    if save_to_disk:
        if project_info.dataset_format == "parquet":
            flattened_df.to_parquet(
                project_info.feature_set_path / "flattened_features.parquet",
            )
        elif project_info.dataset_format == "csv":
            flattened_df.to_csv(
                project_info.feature_set_path / "flattened_features.csv",
            )

    return flattened_df


if __name__ == "__main__":
    generate_flattened_features(save_to_disk=True)
