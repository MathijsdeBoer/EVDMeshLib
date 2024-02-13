from math import pi

import torch
from torch import nn


class MeanAbsoluteAngularError(nn.Module):
    def __init__(
        self,
        x_range: tuple[float, float] = (0.0, 2 * pi),
        y_range: tuple[float, float] = (0.0, pi),
    ) -> None:
        super().__init__()

        self._x_range = x_range
        self._y_range = y_range

        self._x_multiplier = (self._x_range[1] - self._x_range[0]) + self._x_range[0]
        self._y_multiplier = (self._y_range[1] - self._y_range[0]) + self._y_range[0]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self,
        x_range: tuple[float, float] = (0.0, 2 * pi),
        y_range: tuple[float, float] = (0.0, pi),
    ) -> None:
        super().__init__()

        self._x_range = x_range
        self._y_range = y_range

        self._x_multiplier = (self._x_range[1] - self._x_range[0]) + self._x_range[0]
        self._y_multiplier = (self._y_range[1] - self._y_range[0]) + self._y_range[0]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true

        # Both x and y are in range [0, 1], so we can just multiply by the range
        diff[:, 1] *= self._x_multiplier
        diff[:, 0] *= self._y_multiplier

        return torch.mean(diff * diff)

    def __repr__(self) -> str:
        return f"MeanSquaredAngularError(x_range={self._x_range}, y_range={self._y_range})"

    def __str__(self) -> str:
        return "MeanSquaredAngularError"
