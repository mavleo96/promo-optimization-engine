import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from .base_module import BaseModuleClass


class BaselineLayer(BaseModuleClass):
    """
    Baseline layer of the model.

    This layer is responsible for calculating the baseline sales value.
    """

    def __init__(self, hier_shape: int, baseline_init: NDArray):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"
        assert baseline_init.shape == (1, hier_shape), "Baseline init shape mismatch"

        self.base_index_rate = 0.001
        self.start_ratio = 0.3

        self.baseline_intercept = self.create_var(
            (1, hier_shape), init=self.start_ratio * baseline_init, activation="sigmoid"
        )
        self.baseline_weight1, self.baseline_weight1_hier = self.create_hier_var(
            (1, hier_shape), 1, activation="tanh"
        )
        self.baseline_weight2, self.baseline_weight2_hier = self.create_hier_var(
            (1, hier_shape), 1, activation="tanh"
        )

    def forward(self, time_index: torch.Tensor, sales_lag: torch.Tensor) -> torch.Tensor:
        # Calculate the baseline using the linear equation

        # Baseline sales is defined as a linear function of time and sales_lag
        # Baseline = intercept + slope1 * time_index + slope2 * sales_lag
        baseline = (
            self.baseline_intercept.apply_activation()
            + (
                self.baseline_weight1.apply_activation()
                + self.baseline_weight1_hier.apply_activation()
            )
            * (time_index * self.base_index_rate)
            + (
                self.baseline_weight2.apply_activation()
                + self.baseline_weight2_hier.apply_activation()
            )
            * sales_lag
        )
        return baseline


class MixedEffectLayer(BaseModuleClass):
    """
    Mixed effect layer of the model.

    This layer is responsible for calculating the mixed effect / perturbation
    to the baseline sales value and the ROI multiplier / perturbation to the
    discount uplift.
    """

    def __init__(self, hier_shape: int, n_macro: int):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"
        assert n_macro > 0, "Number of macro variables must be positive"

        self.me_mult_param, self.me_mult_param_hier = self.create_hier_var(
            (1, hier_shape, n_macro), 1, activation="tanh"
        )
        self.roi_mult_param, self.roi_mult_param_hier = self.create_hier_var(
            (1, hier_shape, n_macro), 1, activation="tanh"
        )

    def forward(self, macro: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Mixed effect is defined as a non-linear function of the macro variables
        # ME = 1 + tanh(param * variable)
        mixed_effect = 1 + F.tanh(
            (self.me_mult_param.apply_activation() + self.me_mult_param_hier.apply_activation())
            * macro
        )
        roi_mult = 1 + F.tanh(
            (self.roi_mult_param.apply_activation() + self.roi_mult_param_hier.apply_activation())
            * mixed_effect
        )
        return mixed_effect, roi_mult


class DiscountLayer(BaseModuleClass):
    """
    Discount layer of the model.

    This layer is responsible for calculating the discount uplift.
    """

    def __init__(self, hier_shape: int, n_types: int):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"
        assert n_types > 0, "Number of types must be positive"

        self.slope, self.slope_hier = self.create_hier_var(
            (1, hier_shape, n_types), 1, activation="sigmoid"
        )

    def forward(self, discount: torch.Tensor) -> torch.Tensor:
        # Discount uplift is defined as a linear function of the discount
        uplift = (self.slope.apply_activation() + self.slope_hier.apply_activation()) * discount
        return uplift


class VolumeConversion(BaseModuleClass):
    """
    Volume conversion layer of the model.

    This layer is responsible for converting the net revenue to volume.
    """

    def __init__(self, hier_shape: int):
        super().__init__()
        assert hier_shape > 0, "Hierarchy shape must be positive"

        self.slope = self.create_var((1, hier_shape), activation="sigmoid")
        self.intercept = self.create_var((1, hier_shape), activation="tanh")

    def forward(self, sales: torch.Tensor) -> torch.Tensor:
        # Volume conversion is defined as a linear function of the net revenue
        volume = self.slope.apply_activation() * sales + self.intercept.apply_activation()
        return volume
