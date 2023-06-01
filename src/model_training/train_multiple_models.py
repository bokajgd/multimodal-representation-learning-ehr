"""Script to train multiple modelS based on config."""
from pathlib import Path

import pandas as pd
from full_config import FullConfigSchema
from get_search_space import SearchSpaceInferrer
from process_manager_setup import setup
from trainer_spawner import spawn_trainers


def main(
    cfg: FullConfigSchema,
):
    """Main."""
    # Load dataset without dropping any rows for inferring
    # which look distances to grid search over

    train_dataset_name = "train_binary_"

    data_path = cfg.data.dir

    train_df = pd.read_csv(
        [file for file in data_path.iterdir() if train_dataset_name in file.name][0],
    )

    train_df[cfg.data.col_name.pred_timestamp] = pd.to_datetime(
        train_df[cfg.data.col_name.pred_timestamp],
    )

    trainer_specs = SearchSpaceInferrer(
        cfg=cfg,
        train_df=train_df,
        model_names=["xgboost", "logistic-regression"],
    ).get_trainer_specs()

    spawn_trainers(
        cfg=cfg,
        config_file_name=CONFIG_FILE_NAME,
        trainer_specs=trainer_specs,
        train_single_model_file_path=Path(
            "/Users/jakobgrohn/Desktop/Cognitive_Science/Masters_Thesis/multimodal-representation-learning-ehr/src/model_training/train_full_model.py",
        ),
    )


if __name__ == "__main__":
    CONFIG_FILE_NAME = "default_config.yaml"

    cfg = setup(
        config_file_name=CONFIG_FILE_NAME,
        application_config_dir_relative_path="../../src/config/",
    )

    main(cfg=cfg)
