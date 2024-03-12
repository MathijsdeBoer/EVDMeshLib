"""
Point regressor model.
"""
from math import ceil, log2, prod
from typing import Any, Callable, Sequence

import optuna
import torch
from loguru import logger
from monai.networks.blocks import ConvDenseBlock, Convolution
from monai.networks.layers import Act, Flatten, Norm, Reshape, get_act_layer
from monai.networks.nets import Regressor
from torch import nn

from evdplanner.network.training import OptimizableModel


class ParallelConcat(nn.Module):
    """
    Concatenates the outputs of multiple modules in parallel.
    """

    def __init__(self, modules: Sequence[nn.Module], cat_dim: int = 1) -> None:
        """
        Initializes the ParallelConcat module.

        Parameters
        ----------
        modules : Sequence[nn.Module]
            The modules to concatenate.
        cat_dim : int, optional
            The dimension to concatenate along, by default 1.
        """
        super().__init__()
        self.cat_dim = cat_dim

        for idx, module in enumerate(modules):
            self.add_module(f"parallel_cat_{idx}", module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The concatenated output tensor.
        """
        outputs = []
        for module in self.children():
            outputs.append(module(x))
        return torch.cat(outputs, dim=self.cat_dim)


class PointRegressor(Regressor, OptimizableModel):
    """
    A point regressor model that predicts keypoints from a set of input maps.
    """

    def __init__(
        self,
        maps: list[str],
        keypoints: list[str],
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act: nn.Module | Callable | str = Act.PRELU,
        final_act: nn.Module | Callable | str | None = None,
        norm: nn.Module | Callable | str = Norm.INSTANCE,
        dropout: float | None = None,
        bias: bool = True,
        final_bias: float | tuple[float, float] | Sequence[tuple[float, float]] | None = None,
    ) -> None:
        """
        Initializes the PointRegressor model.

        Parameters
        ----------
        maps : list[str]
            The names of the input maps.
        keypoints : list[str]
            The names of the keypoints to predict.
        in_shape : Sequence[int]
            The shape of the input tensor.
        out_shape : Sequence[int]
            The shape of the output tensor.
        channels : Sequence[int]
            The number of channels in each layer.
        strides : Sequence[int]
            The stride of each layer.
        kernel_size : Sequence[int] | int, optional
            The kernel size of the convolutional layers, by default 3.
        num_res_units : int, optional
            The number of residual units in each dense block, by default 2.
        act : nn.Module | Callable | str, optional
            The activation function to use, by default Act.PRELU.
        final_act : nn.Module | Callable | str | None, optional
            The final activation function to use, by default None.
        norm : nn.Module | Callable | str, optional
            The normalization function to use, by default Norm.INSTANCE.
        dropout : float | None, optional
            The dropout rate to use, by default None.
        bias : bool, optional
            Whether to use bias in the convolutional layers, by default True.
        final_bias : float | tuple[float, float] | Sequence[tuple[float, float]] | None, optional
            The bias to use in the final linear layer, by default None.
        """
        logger.debug(f"Creating PointRegressor with in_shape={in_shape}, out_shape={out_shape}")
        self.final_bias = final_bias
        logger.debug(f"Using final_bias={self.final_bias}")

        super().__init__(
            in_shape=in_shape,
            out_shape=out_shape,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        self.maps = maps
        self.keypoints = keypoints
        logger.debug(f"Using maps={self.maps}, keypoints={self.keypoints}")

        self.final_act = final_act
        if isinstance(self.final_act, str):
            self.final_act = get_act_layer(self.final_act)

    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> nn.Sequential:
        d_out = out_channels - in_channels
        dilations = [1, 2, 4]
        d_channels = [d_out // 3, d_out // 3, d_out // 3 + d_out % 3]

        dense_block = ConvDenseBlock(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            channels=d_channels,
            dilations=dilations,
            kernel_size=self.kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )

        convolution = Convolution(
            spatial_dims=self.dimensions,
            in_channels=out_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )

        return nn.Sequential(dense_block, convolution)

    def _get_final_layer(self, in_shape: Sequence[int]) -> nn.Sequential:
        point_paths = []

        for i in range(self.out_shape[0]):
            convolution = Convolution(
                spatial_dims=self.dimensions,
                in_channels=in_shape[0],
                out_channels=in_shape[0] * 2,
                strides=2,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                conv_only=True,
            )

            linear = nn.Linear(
                in_features=prod(in_shape) // 2,
                out_features=self.out_shape[1],
            )

            if self.final_bias:
                if isinstance(self.final_bias, (int, float)):
                    logger.debug(f"Setting final_bias={self.final_bias}")
                    linear.bias.data.fill_(self.final_bias)
                elif isinstance(self.final_bias, Sequence):
                    if len(self.final_bias) == self.out_shape[1]:
                        logger.debug(f"Setting final_bias={self.final_bias}")
                        linear.bias.data = nn.Parameter(torch.tensor(self.final_bias))
                    elif len(self.final_bias) == self.out_shape[0]:
                        logger.debug(f"Setting output layer {i} final_bias={self.final_bias[i]}")
                        linear.bias.data = nn.Parameter(torch.tensor(self.final_bias[i]))
                    else:
                        msg = (
                            "final_bias must have the same length as the number of keypoints,"
                            " the number of dimensions, or be a single float."
                        )
                        logger.error(msg)
                        raise ValueError(msg)
                else:
                    msg = "final_bias must be a float, tuple of floats, or a sequence of tuples of floats."
                    logger.error(msg)
                    raise ValueError(msg)
            point_paths.append(nn.Sequential(convolution, Flatten(), linear))

        return nn.Sequential(ParallelConcat(point_paths), Reshape(*self.out_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = super().forward(x)
        if self.final_act:
            x = self.final_act(x)
        return x

    @classmethod
    def get_optuna_parameters(
        cls: type["PointRegressor"], optuna_trial: optuna.Trial, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Get the parameters for the model from an Optuna trial.

        Parameters
        ----------
        optuna_trial : optuna.Trial
            The Optuna trial to get the parameters from.

        Returns
        -------
        dict[str, Any]
            The parameters for the model.
        """
        match kwargs["anatomy"]:
            case "skin":
                max_depth = int(ceil(log2(kwargs["resolution"]) / 2))

        params: dict[str, Any] = {
            "optimizer": optuna_trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "optimizer_args": {
                "lr": optuna_trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                "weight_decay": optuna_trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
            },
            "filters": optuna_trial.suggest_categorical("filters", [2**i for i in range(3, 6)]),
            "max_filters": optuna_trial.suggest_categorical(
                "max_filters", [2**i for i in range(6, 9)]
            ),
            "num_blocks": optuna_trial.suggest_int("num_blocks", 1, max_depth),
            "num_residual_units": optuna_trial.suggest_int("num_residual_units", 0, 4),
            "act_fn": optuna_trial.suggest_categorical(
                "act_fn",
                [
                    Act.RELU,
                    Act.PRELU,
                    Act.LEAKYRELU,
                    Act.SWISH,
                    Act.MISH,
                ],
            ),
            "norm_fn": optuna_trial.suggest_categorical(
                "norm_fn",
                [
                    Norm.INSTANCE,
                    Norm.BATCH,
                ],
            ),
            "dropout": optuna_trial.suggest_float("dropout", 0.0, 0.75),
            "final_act_fn": optuna_trial.suggest_categorical(
                "final_act_fn",
                [
                    None,
                    Act.RELU,
                    Act.SIGMOID,
                    Act.TANH,
                ],
            ),
            "loss_fn": optuna_trial.suggest_categorical(
                "loss_fn",
                [
                    "mse",
                    "mae",
                    "msae",
                    "maae",
                ],
            ),
        }

        match params["optimizer"]:
            case "adam":
                params["optimizer_args"]["betas"] = (
                    optuna_trial.suggest_float("beta1", 0.001, 0.999),
                    optuna_trial.suggest_float("beta2", 0.5, 0.999),
                )
                params["optimizer_args"]["eps"] = optuna_trial.suggest_float(
                    "eps", 1e-8, 1e-1, log=True
                )
            case "sgd":
                params["optimizer_args"]["nesterov"] = optuna_trial.suggest_categorical(
                    "nesterov",
                    [
                        True,
                        False,
                    ],
                )

                if params["optimizer_args"]["nesterov"]:
                    params["optimizer_args"]["momentum"] = optuna_trial.suggest_float(
                        "momentum", 0.001, 0.999
                    )
                    params["optimizer_args"]["dampening"] = 0.0
                else:
                    params["optimizer_args"]["momentum"] = optuna_trial.suggest_float(
                        "momentum", 0.0, 0.999
                    )
                    params["optimizer_args"]["dampening"] = optuna_trial.suggest_float(
                        "dampening", 0.0, 0.999
                    )

        return params

    @classmethod
    def from_optuna_parameters(
        cls: type["PointRegressor"], parameters: dict[str, any], **kwargs
    ) -> "PointRegressor":
        """
        Create a PointRegressor from Optuna parameters.

        Parameters
        ----------
        parameters : dict[str, any]
            The parameters to use.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        PointRegressor
            The PointRegressor model.
        """
        filters = tuple(
            min(parameters["filters"] * 2**i, parameters["max_filters"])
            for i in range(parameters["num_blocks"])
        )

        result = PointRegressor(
            maps=kwargs["maps"],
            keypoints=kwargs["keypoints"],
            in_shape=kwargs["in_shape"],
            out_shape=kwargs["out_shape"],
            channels=filters,
            strides=(2,) * parameters["num_blocks"],
            kernel_size=3,
            num_res_units=parameters["num_residual_units"],
            act=parameters.get("act_fn", Act.PRELU),
            final_act=parameters.get("final_act_fn", None),
            norm=parameters.get("norm_fn", Norm.INSTANCE),
            dropout=parameters["dropout"],
            bias=True,
            final_bias=kwargs.get("final_bias", None),
        )

        return result

    @staticmethod
    def loggable_parameters() -> list[str]:
        """
        Get the parameters that can be logged.

        Returns
        -------
        list[str]
            The parameters that can be logged.
        """
        return [
            "filters",
            "max_filters",
            "num_blocks",
            "num_residual_units",
            "act_fn",
            "final_act_fn",
            "norm_fn",
            "dropout",
        ]

    @property
    def log_name(self) -> str:
        """
        Get the name to use for logging.

        Returns
        -------
        str
            The name to use for logging.
        """
        return "point_regressor"
