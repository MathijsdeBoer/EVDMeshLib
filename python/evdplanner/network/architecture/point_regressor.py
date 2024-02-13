from math import prod
from typing import Any, Sequence

import optuna
import torch
import torch.nn as nn
from evdplanner.network.training.optimizable_model import OptimizableModel
from monai.networks.blocks import ConvDenseBlock, Convolution
from monai.networks.layers import Act, Flatten, Norm, Reshape
from monai.networks.nets import Regressor


class ParallelConcat(nn.Module):
    def __init__(self, modules: Sequence[nn.Module], cat_dim: int = 1) -> None:
        super().__init__()
        self.cat_dim = cat_dim

        for idx, module in enumerate(modules):
            self.add_module(f"parallel_cat_{idx}", module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for module in self.children():
            outputs.append(module(x))
        return torch.cat(outputs, dim=self.cat_dim)


class PointRegressor(Regressor, OptimizableModel):
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

        for _ in range(self.out_shape[0]):
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

            point_paths.append(nn.Sequential(convolution, Flatten(), linear))

        return nn.Sequential(ParallelConcat(point_paths), Reshape(*self.out_shape))

    @classmethod
    def get_optuna_parameters(cls, optuna_trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {
            "optimizer": optuna_trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "optimizer_args": {
                "lr": optuna_trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                "weight_decay": optuna_trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            },
            "filters": optuna_trial.suggest_int("filters", 16, 64, log=True),
            "max_filters": optuna_trial.suggest_int("max_filters", 16, 256),
            "num_blocks": optuna_trial.suggest_int("num_blocks", 2, 4),
            "num_residual_units": optuna_trial.suggest_int("num_residual_units", 0, 3),
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
                    Act.SIGMOID,
                    Act.TANH,
                ],
            ),
            "loss_fn": optuna_trial.suggest_categorical(
                "loss_fn",
                [
                    "mse",
                    "mae",
                    "msae" "maae",
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
        cls, parameters: dict[str, any], **kwargs: dict[str, Any]
    ) -> "PointRegressor":
        filters = tuple(
            min(parameters["filters"] * 2**i, parameters["max_filters"])
            for i in range(parameters["num_blocks"])
        )

        result = PointRegressor(
            in_shape=kwargs["in_shape"],
            out_shape=kwargs["out_shape"],
            channels=filters,
            strides=(2,) * parameters["num_blocks"],
            kernel_size=3,
            num_res_units=parameters["num_residual_units"],
            act=parameters["act_fn"] or Act.PRELU,
            norm=parameters["norm_fn"] or Norm.INSTANCE,
            dropout=parameters["dropout"],
            bias=True,
        )

        return result

    @classmethod
    def loggable_parameters(cls) -> list[str]:
        return [
            "filters",
            "max_filters",
            "num_blocks",
            "num_residual_units",
            "act_fn",
            "norm_fn",
            "dropout",
        ]
