"""Train a single model and evaluate it."""
from pathlib import Path
from typing import Optional

import pandas as pd
from col_name_inference import get_col_names
from data_schema import ColumnNamesSchema
from dataclasses_schemas import EvalDataset
from full_config import FullConfigSchema
from model_evaluator import ModelEvaluator
from model_pipeline import create_model_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from utils import PROJECT_ROOT


def get_eval_dir() -> Path:
    """Get the directory to save evaluation results to."""

    eval_dir_path = PROJECT_ROOT / "tests" / "test_eval_results"

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

    eval_dataset = train_validate(
        cfg=cfg,
        train=train_datasets,
        val=val_datasets,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
    )

    return eval_dataset


def train_model(
    cfg,
    override_output_dir: Optional[Path] = None,
) -> float:
    """Train a single model and evaluate it."""

    data_path = cfg.data.dir

    train_datasets = pd.read_csv(
        data_path / [file for file in data_path.iterdir() if "train" in file.name][0],
    )

    val_datasets = pd.read_csv(
        data_path / [file for file in data_path.iterdir() if "val" in file.name][0],
    )

    eval_dir_path = get_eval_dir(cfg)

    pipe = create_model_pipeline(cfg)

    outcome_col_name, train_col_names = get_col_names(cfg, train_datasets)

    eval_dataset = train_and_predict(
        cfg=cfg,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
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
