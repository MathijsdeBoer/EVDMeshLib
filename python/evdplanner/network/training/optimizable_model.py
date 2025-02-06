"""
The abstract class for the models that can be optimized using optuna.
"""
from abc import ABC, abstractmethod
from typing import Any

import optuna
from torch import nn


class OptimizableModel(ABC, nn.Module):
    """
    Abstract class for the models that can be optimized using optuna.

    Attributes
    ----------
    maps: list[str]
        List of the maps that the model can use.
    keypoints: list[str]
        List of the keypoints that the model can use.
    """

    maps: list[str]
    keypoints: list[str]

    @classmethod
    @abstractmethod
    def get_optuna_parameters(
        cls: type["OptimizableModel"], optuna_trial: optuna.Trial, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get the parameters for the model from the optuna trial.

        Parameters
        ----------
        optuna_trial : optuna.Trial
            The optuna trial to get the parameters from.

        Returns
        -------
        dict[str, any]
            The parameters for the model.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_optuna_parameters(
        cls: type["OptimizableModel"], parameters: dict[str, Any], **kwargs
    ) -> "OptimizableModel":
        """
        Create a model from the parameters.

        Parameters
        ----------
        parameters : dict[str, any]
        kwargs : any

        Returns
        -------
        OptimizableModel
            The model created from the parameters.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def loggable_parameters() -> list[str]:
        """
        Get the parameters that can be logged.

        Returns
        -------
        list[str]
            The parameters that can be logged.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def log_name(self) -> str:
        """
        Get the name of the model for logging.

        Returns
        -------
        str
            The name of the model for logging.
        """
        raise NotImplementedError
