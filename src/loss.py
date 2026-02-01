import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import EPS
from .utils import tensor_gather

MSE_LAMBDA = 0.01
VOL_LAMBDA = 1.0
REG_LAMBDA = 0.0
HIER_LAMBDA = 0.001


class HierarchicalRegularizationLoss(nn.Module):
    """
    This loss function is used to regularize the hierarchical variables.

    Loss Components:
    - L2 regularization of the global variables
    - L2 regularization of the hierarchical variables
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, hier_params: nn.ParameterList, global_params: nn.ParameterList
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l2_loss = sum(i.square().sum() for i in global_params)
        hier_loss = sum(i.square().mean() for i in hier_params)

        total_loss = HIER_LAMBDA * hier_loss + REG_LAMBDA * l2_loss
        return total_loss, {
            "l2_reg_loss": l2_loss.item(),
            "hier_reg_loss": hier_loss.item(),
        }


class RegressionLoss(nn.Module):
    """
    This loss function is used to learn the hierarchical regression model.

    Loss Components:
    - Smooth loss of the sales prediction
    - Smooth loss of the volume prediction
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Calculate losses with reduction='none' to apply mask manually
        def smooth_loss(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (F.l1_loss(y_hat, y, reduction="none") * mask).mean() + MSE_LAMBDA * (
                F.mse_loss(y_hat, y, reduction="none") * mask
            ).mean()

        sales_loss = smooth_loss(y_hat, y, mask)
        vol_loss = smooth_loss(y_vol_hat, y_vol, mask)

        total_loss = sales_loss + VOL_LAMBDA * vol_loss

        return total_loss, {
            "sales_smooth_loss": sales_loss.item(),
            "volume_smooth_loss": vol_loss.item(),
        }


ROI_LAMBDA = 10
NEGATIVE_DISCOUNT_LAMBDA = 1000
BRAND_CONSTRAINT_LAMBDA = 1000
PACK_CONSTRAINT_LAMBDA = 1000
PRICE_SEGMENT_CONSTRAINT_LAMBDA = 1000
VOLUME_VARIATION_CONSTRAINT_LAMBDA = 1000


class ROILoss(nn.Module):
    """
    This loss function is used to optimize the ROI of the promotions.

    Loss Components:
    - ROI
    - Increase in Sales
    - Negative Discount
    - Brand Constraint
    - Pack Constraint
    - Price Segment Constraint
    - Volume Variation Constraint
    """

    def __init__(
        self,
        constraint_tensors: dict[str, torch.Tensor],
        gather_indices: dict[str, torch.Tensor],
    ):
        super().__init__()

        self.register_buffer("brand_constraint", constraint_tensors["brand"])
        self.register_buffer("pack_constraint", constraint_tensors["pack"])
        self.register_buffer("price_segment_constraint", constraint_tensors["price_segment"])
        self.register_buffer("volume_variation_constraint", constraint_tensors["volume_variation"])

        self.register_buffer("brand_gather_indices", gather_indices["brand"])
        self.register_buffer("pack_gather_indices", gather_indices["pack"])
        self.register_buffer("price_segment_gather_indices", gather_indices["price_segment"])

    def forward(
        self,
        discount_spend: torch.Tensor,
        opt_sales: torch.Tensor,
        init_sales: torch.Tensor,
        opt_vol: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        roi = (opt_sales.sum() - init_sales.sum()) / (discount_spend.sum() + EPS)
        nr_increase = opt_sales.sum() - init_sales.sum()
        negative_discount_loss = F.relu(-discount_spend).sum()

        # Constrain Loss
        discount_brand = tensor_gather(discount_spend, self.brand_gather_indices, dim=1).sum(2)  # type: ignore
        discount_pack = tensor_gather(discount_spend, self.pack_gather_indices, dim=1).sum(2)  # type: ignore
        discount_price_segment = tensor_gather(
            discount_spend,
            self.price_segment_gather_indices,
            dim=1,  # type: ignore
        ).sum(2)

        brand_constraint_loss = F.relu(discount_brand - self.brand_constraint).sum()  # type: ignore
        pack_constraint_loss = F.relu(discount_pack - self.pack_constraint).sum()  # type: ignore
        price_segment_constraint_loss = F.relu(
            discount_price_segment - self.price_segment_constraint  # type: ignore
        ).sum()

        # Volume Variation Loss
        lower_constraint = self.volume_variation_constraint[[0]]  # type: ignore
        upper_constraint = self.volume_variation_constraint[[1]]  # type: ignore
        volume_variation_loss = F.relu(opt_vol - opt_vol * upper_constraint).sum()
        volume_variation_loss += F.relu(opt_vol * lower_constraint - opt_vol).sum()

        loss = (
            -nr_increase
            - ROI_LAMBDA * roi
            + NEGATIVE_DISCOUNT_LAMBDA * negative_discount_loss
            + BRAND_CONSTRAINT_LAMBDA * brand_constraint_loss
            + PACK_CONSTRAINT_LAMBDA * pack_constraint_loss
            + PRICE_SEGMENT_CONSTRAINT_LAMBDA * price_segment_constraint_loss
            + VOLUME_VARIATION_CONSTRAINT_LAMBDA * volume_variation_loss
        )
        return loss, {
            "optimized_roi": roi.item(),
            "nr_increase": nr_increase.item(),
            "negative_discount_loss": negative_discount_loss.item(),
            "brand_constraint_loss": brand_constraint_loss.item(),
            "pack_constraint_loss": pack_constraint_loss.item(),
            "price_segment_constraint_loss": price_segment_constraint_loss.item(),
            "volume_variation_loss": volume_variation_loss.item(),
        }
