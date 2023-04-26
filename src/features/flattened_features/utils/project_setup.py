"""Setup for the project.

Code adapted from GitHub repository 'psycop-feature-generation'
"""
import logging
import time
from pathlib import Path
from typing import Literal

from timeseriesflattener.feature_spec_objects import (
    BaseModel,
)

log = logging.getLogger(__name__)


class Prefixes(BaseModel):
    """Prefixes for feature specs."""

    predictor: str = "pred"
    outcome: str = "outc"
    eval: str = "eval"


class ColNames(BaseModel):
    """Column names for feature specs."""

    timestamp = "timestamp"
    id = "patient_id"


class ProjectInfo(BaseModel):
    """Collection of project info."""

    project_path: Path
    feature_set_path: Path
    feature_set_prefix: str
    dataset_format: Literal["parquet", "csv"] = "csv"
    prefix: Prefixes = Prefixes()
    col_names: ColNames = ColNames()

    def __init__(self, **data):
        super().__init__(**data)

        # Iterate over each attribute. If the attribute is a Path, create it if it does not exist.
        for attr in self.__dict__:
            if isinstance(attr, Path):
                attr.mkdir(exist_ok=True, parents=True)


def create_feature_set_path(
    proj_path: Path,
    feature_set_id: str,
) -> Path:
    """Create save directory.

    Args:
        proj_path (Path): Path to project.
        feature_set_id (str): Feature set id.

    Returns:
        Path: Path to sub directory.
    """

    # Split and save to disk
    # Create directory to store all files related to this run
    save_dir = proj_path / "data" / "feature_sets" / feature_set_id

    save_dir.mkdir(exist_ok=True, parents=True)

    return save_dir


def get_project_info(
) -> ProjectInfo:
    """Setup for main.

    Args:
        project_name (str): Name of project.
    Returns:
        tuple[Path, str]: Tuple of project path, and feature_set_id
    """
    log.info("Setting up project")
    proj_path = Path(__file__).resolve().parents[4]


    feature_set_id = f"multimodal_rep_learning_ehr_features_{time.strftime('%Y_%m_%d_%H_%M')}"

    feature_set_path = create_feature_set_path(
        feature_set_id=feature_set_id,
        proj_path=proj_path,
    )

    return ProjectInfo(
        project_path=proj_path,
        feature_set_path=feature_set_path,
        feature_set_prefix=feature_set_id,
    )