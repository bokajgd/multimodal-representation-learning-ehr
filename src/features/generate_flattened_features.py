"""Main feature generation."""
import logging

import pandas as pd
from feature_specification.specify_features import FeatureSpecifier
from static_and_flattened_features.loaders.utils import DATA_PATH
from utils.flatten_dataset import create_flattened_dataset
from utils.project_setup import get_project_info
from utils.utils import add_age, remove_outliers

log = logging.getLogger(__name__)


def generate_flattened_features(
    save_to_disk: bool = False,
    min_set_for_debug: bool = False,
) -> pd.DataFrame:
    """Main function for generating a feature dataset."""

    predictions_times_df_path = DATA_PATH / "misc" / "cohort_with_prediction_times.csv"
    prediction_times_df = pd.read_csv(predictions_times_df_path)

    # Keep only the last 1000 rows
    prediction_times_df = prediction_times_df.iloc[5000:].reset_index(drop=True)

    # Convert to datetime
    prediction_times_df["timestamp"] = pd.to_datetime(
        prediction_times_df["timestamp"],
    )

    project_info = get_project_info()

    feature_specs = FeatureSpecifier(
        project_info=project_info,
        min_set_for_debug=min_set_for_debug,
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=prediction_times_df,
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )

    # Remove outliers
    flattened_df = remove_outliers(flattened_df)

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
