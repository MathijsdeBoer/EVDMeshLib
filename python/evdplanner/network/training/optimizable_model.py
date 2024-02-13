from abc import ABC, abstractmethod
from typing import Any

import optuna
from torch import nn


class OptimizableModel(ABC):
    @classmethod
    @abstractmethod
    def get_optuna_parameters(cls, optuna_trial: optuna.Trial) -> dict[str, any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_optuna_parameters(
        cls, parameters: dict[str, any], **kwargs: dict[str, Any]
    ) -> nn.Module:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def loggable_parameters(cls) -> list[str]:
        raise NotImplementedError
