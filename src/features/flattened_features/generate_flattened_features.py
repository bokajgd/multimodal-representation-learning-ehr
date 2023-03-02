"""Main feature generation."""
from feautre_specification.specify_features import FeatureSpecifier
from loaders.load_admissions import load_emergency_admissions

from utils.flatten_dataset import (
    create_flattened_dataset,
)
from utils.project_setup import (
    get_project_info,
)


def main():
    """Main function for loading, generating and evaluating a flattened
    dataset."""
    feature_specs = FeatureSpecifier(
        project_info=project_info,
    ).get_feature_specs()

    flattened_df = create_flattened_dataset(
        feature_specs=feature_specs,
        prediction_times_df=load_emergency_admissions(timestamps_only=True),
        drop_pred_times_with_insufficient_look_distance=False,
        project_info=project_info,
    )
    print('hi')

if __name__ == "__main__":

    project_info = get_project_info(
    )
    
    main()