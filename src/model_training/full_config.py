"""Full configuration schema."""
from typing import Optional

from basemodel import BaseModel
from data_schema import DataSchema
from debug import DebugConfSchema
from model import ModelConfSchema
from preprocessing import PreprocessingConfigSchema
from project import ProjectSchema
from train import TrainConfSchema


class FullConfigSchema(BaseModel):
    """A recipe for a full configuration object."""

    project: ProjectSchema
    data: DataSchema
    preprocessing: PreprocessingConfigSchema
    model: ModelConfSchema
    train: TrainConfSchema
    debug: Optional[DebugConfSchema]
