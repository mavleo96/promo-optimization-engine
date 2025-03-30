import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union


class BaseModuleClass(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        self.global_var_list = []
        self.hier_var_list = []
        super().__init__(*args, **kwargs)

    def create_hier_var(
        self, shape: Tuple, hier_dim: int, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        assert 0 < hier_dim < len(shape)
        global_shape = (*shape[:hier_dim], 1, *shape[hier_dim + 1 :])

        global_var = torch.zeros(global_shape, requires_grad=True, dtype=dtype)
        hier_var = torch.zeros(shape, requires_grad=True, dtype=dtype)

        self.global_var_list.append(global_var)
        self.hier_var_list.append(hier_var)

        return global_var + hier_var

    def create_var(
        self,
        shape: Tuple,
        dtype: torch.dtype = torch.float64,
        init: Union[np.ndarray, None] = None,
    ) -> torch.Tensor:
        if init is None:
            var = torch.zeros(shape, requires_grad=True, dtype=dtype)
        else:
            var = torch.tensor(init, requires_grad=True, dtype=dtype)
        return var


class BaselineLayer(BaseModuleClass):
    def __init__(self, n_brands: int, baseline_init: np.ndarray):
        super(BaselineLayer, self).__init__()
        assert n_brands > 0, "Number of brands must be positive"
        assert baseline_init.shape == (1, n_brands), "Baseline init shape mismatch"

        self.baseline_intercept = self.create_var((1, n_brands), init=baseline_init)
        self.baseline_weight1 = self.create_hier_var((1, n_brands), 1)
        self.baseline_weight2 = self.create_hier_var((1, n_brands), 1)

    def forward(self, time_index: torch.Tensor, nr_lag: torch.Tensor) -> torch.Tensor:
        # Calculate the baseline using the linear equation
        baseline = (
            self.baseline_intercept
            + self.baseline_weight1 * time_index
            + self.baseline_weight2 * nr_lag
        )
        return baseline


class MixedEffectLayer(BaseModuleClass):
    def __init__(self, n_brands: int, n_macro: int):
        super(MixedEffectLayer, self).__init__()
        assert n_brands > 0, "Number of brands must be positive"
        assert n_macro > 0, "Number of macro variables must be positive"

        self.me_mult = self.create_hier_var((1, n_brands, n_macro), 1)

    def forward(self, macro: torch.Tensor) -> torch.Tensor:
        mixed_effect = 1 + F.tanh(self.me_mult * macro)
        return mixed_effect


class DiscountLayer(BaseModuleClass):
    def __init__(self, n_brands: int, n_types: int):
        super(DiscountLayer, self).__init__()
        assert n_brands > 0, "Number of brands must be positive"
        assert n_types > 0, "Number of types must be positive"

        self.slope = self.create_hier_var((1, n_brands, n_types), 1)

    def forward(self, discount: torch.Tensor) -> torch.Tensor:
        uplift = self.slope * discount
        return uplift


# Add ROI Mults layers
