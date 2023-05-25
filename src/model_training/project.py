"""Project configuration schemas."""
from basemodel import BaseModel


class ProjectSchema(BaseModel):
    """Project configuration."""

    name: str = "masters_thesis"
    seed: int
    gpu: bool
