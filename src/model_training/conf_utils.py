"""Utilities for handling config objects, e.g. load, change format."""
from full_config import FullConfigSchema
from omegaconf import DictConfig, OmegaConf


def convert_omegaconf_to_pydantic_object(
    conf: DictConfig,
    allow_mutation: bool = False,
) -> FullConfigSchema:
    """Converts an omegaconf DictConfig to a pydantic object.

    Args:
        conf (DictConfig): Omegaconf DictConfig
        allow_mutation (bool, optional): Whether to allow mutation of the

    Returns:
        FullConfig: Pydantic object
    """
    conf = OmegaConf.to_container(conf, resolve=True)  # type: ignore
    return FullConfigSchema(**conf, allow_mutation=allow_mutation)  # type: ignore
