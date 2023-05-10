"""Script for training a model based on config.
"""
from pathlib import Path

import hydra
from conf_utils import convert_omegaconf_to_pydantic_object
from full_config import FullConfigSchema
from omegaconf import DictConfig
from train_model_functions import train_model
from utils import PROJECT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT  / "config"



@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="default_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """Main."""

    if not isinstance(cfg, FullConfigSchema):
        cfg = convert_omegaconf_to_pydantic_object(cfg)

    return train_model(cfg=cfg)


if __name__ == "__main__":
    main()