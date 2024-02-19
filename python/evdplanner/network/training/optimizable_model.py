from abc import ABC, abstractmethod

import optuna
from torch import nn


class OptimizableModel(ABC, nn.Module):
    @classmethod
    @abstractmethod
    def get_optuna_parameters(cls, optuna_trial: optuna.Trial) -> dict[str, any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_optuna_parameters(cls, parameters: dict[str, any], **kwargs) -> "OptimizableModel":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loggable_parameters() -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def log_name(self) -> str:
        raise NotImplementedError
