from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MSE_LAMBDA = 0.01
VOL_LAMBDA = 1.0
REG_LAMBDA = 0.0
HIER_LAMBDA = 0.001


class HierarchicalLoss(nn.Module):
    """
    This loss function is used to learn the hierarchical regression model.

    Loss Components:
    - Smooth loss of the sales prediction
    - Smooth loss of the volume prediction
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
        # Calculate losses with reduction='none' to apply mask manually
        smooth_loss = (
            lambda y_hat, y, mask: (F.l1_loss(y_hat, y, reduction="none") * mask).mean()
            + MSE_LAMBDA * (F.mse_loss(y_hat, y, reduction="none") * mask).mean()
        )

        sales_loss = smooth_loss(y_hat, y, mask)
        vol_loss = smooth_loss(y_vol_hat, y_vol, mask)

        l2_loss = torch.tensor([i.square().sum() for i in global_params]).sum()
        # hierarchical loss is mean instead of sum to avoid greater penalty on higher levels
        hier_loss = torch.tensor([i.square().mean() for i in hier_params]).sum()

        total_loss = (
            sales_loss + VOL_LAMBDA * vol_loss + HIER_LAMBDA * hier_loss + REG_LAMBDA * l2_loss
        )

        return total_loss, {
            "sales_loss": sales_loss.item(),
            "vol_loss": vol_loss.item(),
            "l2_loss": l2_loss.item(),
            "hier_loss": hier_loss.item(),
            "total_loss": total_loss.item(),
        }
