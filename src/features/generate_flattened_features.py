"""Feature generation script."""
import logging

import pandas as pd
from feature_specification.full_feature_specification import FullFeatureSpecifier
from feature_specification.saps_ii_feature_specification import SAPSFeatureSpecifier
from static_and_flattened_features.loaders.load_demographics import load_dod
from static_and_flattened_features.loaders.utils import DATA_PATH
from utils.flatten_dataset import create_flattened_dataset
from utils.project_setup import get_project_info
from utils.utils import add_age, remove_outliers

log = logging.getLogger(__name__)


def generate_flattened_features(
    save_to_disk: bool = False,
    min_set_for_debug: bool = True,
    saps_ii: bool = False,
    get_text_features: bool = False,
) -> pd.DataFrame:
    """Main function for generating a feature dataset."""

    # Load prediction times
    predictions_times_df_path = DATA_PATH / "misc" / "cohort_with_prediction_times.csv"
    prediction_times_df = pd.read_csv(predictions_times_df_path).drop(
        columns=["HADM_ID"],
    )

    # Keep only the last 1000 rows
    # prediction_times_df = prediction_times_df.iloc[10000:].reset_index(drop=True)

    # Convert to datetime
    prediction_times_df["timestamp"] = pd.to_datetime(
        prediction_times_df["timestamp"],
    )

    project_info = get_project_info()

    if saps_ii:
        feature_set_prefix = "saps_ii"
        feature_specs = SAPSFeatureSpecifier(
            project_info=project_info,
            min_set_for_debug=min_set_for_debug,
            get_text_specs=get_text_features,
        ).get_feature_specs()
    else:
        feature_set_prefix = "full"
        feature_specs = FullFeatureSpecifier(
            project_info=project_info,
            min_set_for_debug=min_set_for_debug,
            get_text_specs=get_text_features,
        ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=prediction_times_df,
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )

    # remove outliers
    flattened_df = remove_outliers(flattened_df)

    # add age
    flattened_df = add_age(flattened_df)

    # add dod
    dod = load_dod()

    # rename timestamp to eval_dod
    dod = dod.rename(columns={"timestamp": "eval_dod"})

    # merge dod with flattened_df on patient_id
    flattened_df = flattened_df.merge(
        dod[["patient_id", "eval_dod"]],
        on="patient_id",
        how="left",
    )

    if save_to_disk:
        if get_text_features:
            flattened_df.to_csv(
                project_info.feature_set_path
                / f"{feature_set_prefix}_flattened_features_with_text.csv",
            )
        else:
            flattened_df.to_csv(
                project_info.feature_set_path
                / f"{feature_set_prefix}_flattened_features.csv",
            )

    return flattened_df, feature_set_prefix, project_info


if __name__ == "__main__":
    generate_flattened_features(save_to_disk=False)
