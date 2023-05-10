"""Create post split pipeline.""" ""
from typing import Any

from full_config import FullConfigSchema
from model_specs import MODELS
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_model(cfg: FullConfigSchema) -> Any:
    """Instantiate and return a model object based on settings in the config
    file."""
    model_dict: dict[str, Any] = MODELS.get(cfg.model.name)

    model_args = model_dict["static_hyperparameters"]

    training_arguments = cfg.model.args
    model_args.update(training_arguments)

    return model_dict["model"](**model_args)


def create_model_pipeline(cfg: FullConfigSchema) -> Pipeline:
    """Create pipeline.

    Args:
        cfg (DictConfig): Config object

    Returns:
        Pipeline
    """
    steps = []

    steps.append(
        (
            "Imputation",
            SimpleImputer(strategy="mean"),
        ),
    )
    steps.append(
        (
        "feature_selection",
        SelectPercentile(
            mutual_info_classif,
            percentile=50,
            ),
        ),
    )

    mdl = create_model(cfg)
    
    steps.append(("model", mdl))

    return Pipeline(steps)