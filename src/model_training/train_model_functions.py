"""Train a single model and evaluate it."""
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from wasabi import Printer

curr_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

from col_name_inference import get_col_names
from data_schema import ColumnNamesSchema
from dataclasses_schemas import EvalDataset
from full_config import FullConfigSchema
from model_evaluator import ModelEvaluator
from model_pipeline import create_model_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from utils import PROJECT_ROOT


def get_eval_dir() -> Path:
    """Get the directory to save evaluation results to."""

    eval_dir_path = (
        PROJECT_ROOT / "outputs" / "model_outputs" / f"model_outputs_{curr_timestamp}"
    )

    eval_dir_path.mkdir(parents=True, exist_ok=True)

    return eval_dir_path


def create_eval_dataset(
    col_names: ColumnNamesSchema,
    outcome_col_name: str,
    df: pd.DataFrame,
) -> EvalDataset:
    """Create an evaluation dataset object from a dataframe and
    ColumnNamesSchema."""
    # Check if custom attribute exists:
    custom_col_names = col_names.custom_columns

    custom_columns = {}

    if custom_col_names is not None:
        custom_columns = {col_name: df[col_name] for col_name in custom_col_names}

    # Add all eval_ columns to custom_columns attribute
    eval_columns = {
        col_name: df[col_name]
        for col_name in df.columns
        if col_name.startswith("eval_")
    }

    if len(eval_columns) > 0:
        custom_columns.update(eval_columns)

    eval_dataset = EvalDataset(
        ids=df[col_names.id],
        pred_time_uuids=df[col_names.pred_time_uuid],
        y=df[outcome_col_name],
        y_hat_probs=df["y_hat_prob"],
        pred_timestamps=df[col_names.pred_timestamp],
        outcome_timestamps=df[col_names.outcome_timestamp],
        age=df[col_names.age],
        is_female=df[col_names.is_female],
        exclusion_timestamps=df[col_names.exclusion_timestamp]
        if col_names.exclusion_timestamp
        else None,
        custom_columns=custom_columns if len(custom_columns) > 0 else None,
    )

    return eval_dataset


def train_validate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> EvalDataset:
    """Train model on pre-defined train and validation split and return
    evaluation dataset.

    Args:
        cfg (FullConfig): Config object
        train: Training dataset
        val: Validation dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """

    X_train = train[train_col_names]  # pylint: disable=invalid-name
    y_train = train[outcome_col_name]
    X_val = val[train_col_names]  # pylint: disable=invalid-name

    pipe.fit(X_train, y_train)

    y_train_hat_prob = pipe.predict_proba(X_train)[:, 1]
    y_val_hat_prob = pipe.predict_proba(X_val)[:, 1]

    print(
        f"Performance on train: {round(roc_auc_score(y_train, y_train_hat_prob), 3)}",
    )

    df = val
    df["y_hat_prob"] = y_val_hat_prob

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
    )


def stratified_cross_validation(  # pylint: disable=too-many-locals
    cfg: FullConfigSchema,
    pipe: Pipeline,
    train_df: pd.DataFrame,
    train_col_names: list[str],
    outcome_col_name: str,
) -> pd.DataFrame:
    """Performs stratified and grouped cross validation using the pipeline."""
    msg = Printer(timestamp=True)

    X = train_df[train_col_names]  # pylint: disable=invalid-name
    y = train_df[outcome_col_name]  # pylint: disable=invalid-name

    # Create folds
    msg.info("Creating folds")
    msg.info(f"Training on {X.shape[1]} columns and {X.shape[0]} rows")

    folds = StratifiedGroupKFold(n_splits=5).split(
        X=X,
        y=y,
        groups=train_df[cfg.data.col_name.id],
    )

    # Perform CV and get out of fold predictions
    train_df["oof_y_hat"] = np.nan

    for i, (train_idxs, val_idxs) in enumerate(folds):
        msg_prefix = f"Fold {i + 1}"

        msg.info(f"{msg_prefix}: Training fold")

        X_train, y_train = (  # pylint: disable=invalid-name
            X.loc[train_idxs],
            y.loc[train_idxs],
        )  # pylint: disable=invalid-name
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_train)[:, 1]

        msg.info(f"{msg_prefix}: AUC = {round(roc_auc_score(y_train,y_pred), 3)}")

        train_df.loc[val_idxs, "oof_y_hat"] = pipe.predict_proba(X.loc[val_idxs])[
            :,
            1,
        ]

    return train_df


def crossvalidate(
    cfg: FullConfigSchema,
    train: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
) -> EvalDataset:
    """Train model on cross validation folds and return evaluation dataset.

    Args:
        cfg: Config object
        train: Training dataset
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """

    df = stratified_cross_validation(
        cfg=cfg,
        pipe=pipe,
        train_df=train,
        train_col_names=train_col_names,
        outcome_col_name=outcome_col_name,
    )

    df = df.rename(columns={"oof_y_hat": "y_hat_prob"})

    return create_eval_dataset(
        col_names=cfg.data.col_name,
        outcome_col_name=outcome_col_name,
        df=df,
    )


def train_and_predict(
    cfg,
    train_datasets: pd.DataFrame,
    pipe: Pipeline,
    outcome_col_name: str,
    train_col_names: list[str],
    val_datasets: Optional[pd.DataFrame] = None,
):
    """Train model and return evaluation dataset.

    Args:
        cfg: Config object
        train_datasets: Training datasets
        val_datasets: Validation datasets
        pipe: Pipeline
        outcome_col_name: Name of the outcome column
        train_col_names: Names of the columns to use for training

    Returns:
        Evaluation dataset
    """
    if cfg.model.name in ("ebm", "xgboost"):
        pipe["model"].feature_names = train_col_names  # type: ignore

    if val_datasets is not None:
        eval_dataset = train_validate(
            cfg=cfg,
            train=train_datasets,
            val=val_datasets,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )
    else:
        eval_dataset = crossvalidate(
            cfg=cfg,
            train=train_datasets,
            pipe=pipe,
            outcome_col_name=outcome_col_name,
            train_col_names=train_col_names,
        )

    return eval_dataset


def train_model(
    cfg,
    dataset: str = "flattened",
    outcome_col_to_drop: Optional[
        str
    ] = "outc_date_of_death_within_3_days_bool_fallback_0_dichotomous",
    override_output_dir: Optional[Path] = None,
    cross_validate_model: bool = False,
) -> float:
    """Train a single model and evaluate it.

    Args:
        cfg: Config object
        dataset: Dataset to train model on
        outcome_col_to_drop: Outcome column to drop
        override_output_dir: Override output directory
        cross_validate_model: Whether to cross validate the model or not

    Returns:
        AUC score
    """

    data_path = cfg.data.dir

    eval_dir_path = get_eval_dir()

    # Load data
    train_dataset_name = f"train_{dataset}"
    test_dataset_name = f"test_{dataset}"

    train_data = pd.read_csv(
        [file for file in data_path.iterdir() if train_dataset_name in file.name][0],
    ).drop(columns=outcome_col_to_drop)

    val_data = pd.read_csv(
        [file for file in data_path.iterdir() if test_dataset_name in file.name][0],
    ).drop(columns=outcome_col_to_drop)

    if cross_validate_model:
        train_data = pd.concat([train_data, val_data], ignore_index=True)

    pipe = create_model_pipeline(cfg)

    outcome_col_name, train_col_names = get_col_names(cfg, train_data)

    eval_dataset = train_and_predict(
        cfg=cfg,
        train_datasets=train_data,
        val_datasets=val_data if not cross_validate_model else None,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    )

    eval_dir = eval_dir_path if override_output_dir is None else override_output_dir

    roc_auc = ModelEvaluator(
        eval_dir_path=eval_dir,
        cfg=cfg,
        pipe=pipe,
        eval_ds=eval_dataset,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    ).evaluate_and_save_eval_data()
    return roc_auc
