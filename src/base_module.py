from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class ActivatedParameter(nn.Parameter):
    """
    Custom torch parameter class that can return the activated value of the parameter.

    This is needed to ensure that the parameter is registered as a parameter in the model
    while still being able to use the activated version of the parameter in the model.
    """

    def __new__(cls, data, activation: Union[str, None] = None, requires_grad=True):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data, activation: Union[str, None] = None, requires_grad=True):
        super().__init__()
        self.activation = activation

    def apply_activation(self):
        """Applies the stored activation function dynamically."""
        if self.activation == "tanh":
            return torch.tanh(self)
        elif self.activation == "sigmoid":
            return torch.sigmoid(self)
        elif self.activation == "relu":
            return F.relu(self)
        else:
            return self


HierarchicalActivatedParams = Tuple[ActivatedParameter, ActivatedParameter]


class BaseModuleClass(nn.Module):
    """
    Base class for all modules in the model.

    This class provides a base for creating modules that can be used to create
    hierarchical variables and parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_params = nn.ParameterList()
        self.hier_params = nn.ParameterList()

    def create_hier_var(
        self,
        shape: Tuple,
        hier_dim: int,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
    ) -> HierarchicalActivatedParams:
        assert 0 < hier_dim < len(shape)
        global_shape = (*shape[:hier_dim], 1, *shape[hier_dim + 1 :])

        global_var = self.create_var(
            global_shape, dtype=dtype, activation=activation, track=True
        )
        hier_var = self.create_var(
            shape, dtype=dtype, activation=activation, track=False
        )

        self.hier_params.append(hier_var)

        return global_var, hier_var

    def create_var(
        self,
        shape: Tuple,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float32,
        init: Union[NDArray, None] = None,
        track: bool = True,
    ) -> ActivatedParameter:
        if init is None:
            var = ActivatedParameter(
                torch.randn(shape, dtype=dtype, device="cpu"), activation=activation
            )
        else:
            var = ActivatedParameter(
                torch.tensor(init, dtype=dtype, device="cpu"), activation=activation
            )

        if track:
            self.global_params.append(var)

        return var
