import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union
from numpy.typing import NDArray


class BaseModuleClass(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hier_var_list = []
        # self.global_var_list = []

    def create_hier_var(
        self,
        shape: Tuple,
        hier_dim: int,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        assert 0 < hier_dim < len(shape)
        global_shape = (*shape[:hier_dim], 1, *shape[hier_dim + 1 :])

        global_var = self.create_var(global_shape, dtype=dtype, activation=activation)
        hier_var = self.create_var(shape, dtype=dtype, activation=activation)

        # self.global_var_list.append(global_var)
        self.hier_var_list.append(hier_var)

        return global_var + hier_var

    def create_var(
        self,
        shape: Tuple,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
        init: Union[NDArray, None] = None,
    ) -> torch.Tensor:
        if init is None:
            # TODO: need to register as parameter
            # TODO: check why this is float64 instead of float32
            var = nn.Parameter(
                torch.randn(shape, dtype=dtype, device="cpu"), requires_grad=True
            )
        else:
            var = nn.Parameter(
                torch.tensor(init, dtype=dtype, device="cpu"), requires_grad=True
            )
        if activation == "tanh":
            return F.tanh(var)
        elif activation == "sigmoid":
            return F.sigmoid(var)
        else:
            return var


class BaselineLayer(BaseModuleClass):
    def __init__(self, hier_shape: int, baseline_init: NDArray):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"
        assert baseline_init.shape == (1, hier_shape), "Baseline init shape mismatch"

        self.base_index_rate = 0.001
        self.start_ratio = 0.3

        self.baseline_intercept = self.create_var(
            (1, hier_shape), init=self.start_ratio * baseline_init, activation="sigmoid"
        )
        self.baseline_weight1 = self.create_hier_var(
            (1, hier_shape), 1, activation="tanh"
        )
        self.baseline_weight2 = self.create_hier_var(
            (1, hier_shape), 1, activation="tanh"
        )

    def forward(self, time_index: torch.Tensor, nr_lag: torch.Tensor) -> torch.Tensor:
        # Calculate the baseline using the linear equation
        baseline = (
            self.baseline_intercept
            + self.baseline_weight1 * (time_index * self.base_index_rate)
            + self.baseline_weight2 * nr_lag
        )
        return baseline


class MixedEffectLayer(BaseModuleClass):
    def __init__(self, hier_shape: int, n_macro: int):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"
        assert n_macro > 0, "Number of macro variables must be positive"

        self.me_mult_param = self.create_hier_var(
            (1, hier_shape, n_macro), 1, activation="tanh"
        )
        self.roi_mult_param = self.create_hier_var(
            (1, hier_shape, n_macro), 1, activation="tanh"
        )

    def forward(self, macro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_effect = 1 + F.tanh(self.me_mult_param * macro)
        roi_mult = 1 + F.tanh(self.roi_mult_param * mixed_effect)
        return mixed_effect, roi_mult


class DiscountLayer(BaseModuleClass):
    def __init__(self, hier_shape: int, n_types: int):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"
        assert n_types > 0, "Number of types must be positive"

        self.slope = self.create_hier_var(
            (1, hier_shape, n_types), 1, activation="sigmoid"
        )

    def forward(self, discount: torch.Tensor) -> torch.Tensor:
        uplift = self.slope * discount
        return uplift


class VolumeConversion(BaseModuleClass):
    def __init__(self, hier_shape: int):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"

        self.slope = self.create_var((1, hier_shape), activation="sigmoid")
        self.intercept = self.create_var((1, hier_shape), activation="tanh")

    def forward(self, nr: torch.Tensor) -> torch.Tensor:
        volume = self.slope * nr + self.intercept
        return volume
