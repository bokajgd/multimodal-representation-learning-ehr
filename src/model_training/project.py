"""Project configuration schemas."""
from basemodel import BaseModel


class ProjectSchema(BaseModel):
    """Project configuration."""

    name: str = "psycop_model_training"
    seed: int
    gpu: bool
