"""
The implementation of the angular error metrics.
"""
from math import pi

import torch
from torch import nn


class MeanAbsoluteAngularError(nn.Module):
    """
    Mean Absolute Angular Error (MAAE) is a metric used to measure the
    difference between two angles.

    Attributes
    ----------
    x_range : tuple[float, float]
        The range of the x-axis.
    y_range : tuple[float, float]
        The range of the y-axis.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (0.0, 2 * pi),
        y_range: tuple[float, float] = (0.0, pi),
    ) -> None:
        """
        Initializes the MeanAbsoluteAngularError class.

        Parameters
        ----------
        x_range : tuple[float, float]
            The range of the x-axis.
        y_range : tuple[float, float]
            The range of the y-axis.
        """
        super().__init__()

        self._x_range = x_range
        self._y_range = y_range

        self._x_multiplier = (self._x_range[1] - self._x_range[0]) + self._x_range[0]
        self._y_multiplier = (self._y_range[1] - self._y_range[0]) + self._y_range[0]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MeanAbsoluteAngularError class.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_true : torch.Tensor
            The true values.

        Returns
        -------
        torch.Tensor
            The mean absolute angular error.
        """
        abs_diff = torch.abs(y_pred - y_true)

        # Both x and y are in range [0, 1], so we can just multiply by the range
        abs_diff[:, 1] *= self._x_multiplier
        abs_diff[:, 0] *= self._y_multiplier

        return torch.mean(abs_diff)

    def __repr__(self) -> str:
        return f"MeanAbsoluteAngularError(x_range={self._x_range}, y_range={self._y_range})"

    def __str__(self) -> str:
        return "MeanAbsoluteAngularError"


class MeanSquaredAngularError(nn.Module):
    """
    Mean Squared Angular Error (MSAE) is a metric used to measure the
    difference between two angles.

    Attributes
    ----------
    x_range : tuple[float, float]
        The range of the x-axis.
    y_range : tuple[float, float]
        The range of the y-axis.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (0.0, 2 * pi),
        y_range: tuple[float, float] = (0.0, pi),
    ) -> None:
        """
        Initializes the MeanSquaredAngularError class.

        Parameters
        ----------
        x_range : tuple[float, float]
            The range of the x-axis.
        y_range : tuple[float, float]
            The range of the y-axis.
        """
        super().__init__()

        self._x_range = x_range
        self._y_range = y_range

        self._x_multiplier = (self._x_range[1] - self._x_range[0]) + self._x_range[0]
        self._y_multiplier = (self._y_range[1] - self._y_range[0]) + self._y_range[0]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MeanSquaredAngularError class.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_true : torch.Tensor
            The true values.

        Returns
        -------
        torch.Tensor
            The mean squared angular error.
        """
        diff = y_pred - y_true

        # Both x and y are in range [0, 1], so we can just multiply by the range
        diff[:, 1] *= self._x_multiplier
        diff[:, 0] *= self._y_multiplier

        return torch.mean(diff * diff)

    def __repr__(self) -> str:
        return f"MeanSquaredAngularError(x_range={self._x_range}, y_range={self._y_range})"

    def __str__(self) -> str:
        return "MeanSquaredAngularError"
