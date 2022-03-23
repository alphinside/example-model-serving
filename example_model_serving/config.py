from enum import Enum
from functools import lru_cache

from pydantic import BaseSettings


class ModelType(str, Enum):
    svc = "svc"
    decision_tree = "decision_tree"


class Settings(BaseSettings):
    model_type: ModelType

    class Config:
        env_file = ".env.example"


@lru_cache()
def get_settings():
    settings = Settings()
    return settings
