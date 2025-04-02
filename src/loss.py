from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

VOL_LAMBDA = 1
REG_LAMBDA = 0
HIER_LAMBDA = 1e-5


class HierarchicalLoss(nn.Module):
    """
    This loss function is used to learn the hierarchical regression model.

    Loss Components:
    - Mean Squared Error (MSE) of the net revenue prediction
    - Mean Squared Error (MSE) of the volume prediction
    - L2 regularization of the hierarchical variables
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        y_vol_hat: torch.Tensor,
        y_vol: torch.Tensor,
        mask: torch.Tensor,
        hier_params: nn.ParameterList,
        global_params: nn.ParameterList,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Calculate MSE losses with reduction='none' to apply mask manually
        nr_loss = (mse_loss(y_hat, y, reduction="none") * mask).mean()
        vol_loss = (mse_loss(y_vol_hat, y_vol, reduction="none") * mask).mean()

        l2_loss = torch.tensor([i.square().sum() for i in global_params]).sum()
        # hierarchical loss is mean instead of sum to avoid greater penalty on higher levels
        hier_loss = torch.tensor([i.square().mean() for i in hier_params]).sum()

        total_loss = (
            nr_loss
            + VOL_LAMBDA * vol_loss
            + HIER_LAMBDA * hier_loss
            + REG_LAMBDA * l2_loss
        )

        return total_loss, {
            "nr_loss": nr_loss.item(),
            "vol_loss": vol_loss.item(),
            "l2_loss": l2_loss.item(),
            "hier_loss": hier_loss.item(),
            "total_loss": total_loss.item(),
        }
