from typing import Tuple

import lightning as L
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset

from .dataset import Dataset
from .loss import HierarchicalLoss
from .model_components import (
    BaselineLayer,
    DiscountLayer,
    MixedEffectLayer,
    VolumeConversion,
)


class HierarchicalRegressionModel(L.LightningModule):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        # can pass encodings instead of dataset obj
        super().__init__(*args, **kwargs)
        self.n_sku = dataset.n_sku
        self.n_macro = dataset.n_macro
        self.n_discount_type = dataset.n_discount_type
        self.baseline_init = dataset.base_init

        # Initialize layers as nn.ModuleList to ensure proper parameter registration
        self.baseline_layer = BaselineLayer(
            hier_shape=self.n_sku, baseline_init=self.baseline_init
        )
        self.me_layer = MixedEffectLayer(hier_shape=self.n_sku, n_macro=self.n_macro)
        self.discount_layer = DiscountLayer(
            hier_shape=self.n_sku, n_types=self.n_discount_type
        )
        self.convert_to_volume = VolumeConversion(hier_shape=self.n_sku)

        self.global_params = [
            *self.baseline_layer.global_params,
            *self.me_layer.global_params,
            *self.discount_layer.global_params,
            *self.convert_to_volume.global_params,
        ]
        self.hier_params = [
            *self.baseline_layer.hier_params,
            *self.me_layer.hier_params,
            *self.discount_layer.hier_params,
            *self.convert_to_volume.hier_params,
        ]
        self.loss_fn = HierarchicalLoss()

    def forward(self, x: TensorDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, time_index, sales_lag, macro, discount = x

        # Use layers from ModuleList
        baseline = self.baseline_layer(time_index, sales_lag)
        mixed_effect, roi_mult = self.me_layer(macro)
        discount_uplift = self.discount_layer(discount)

        sales_pred = baseline * torch.prod(mixed_effect, dim=2) + torch.sum(
            discount_uplift, dim=2
        ) * torch.prod(roi_mult, dim=2)
        volume_pred = self.convert_to_volume(sales_pred)

        return sales_pred, volume_pred

    def training_step(self, batch: TensorDataset) -> torch.Tensor:
        y, y_vol, mask, _, _, _, _ = batch
        y_hat, y_vol_hat = self(batch)
        loss, loss_components = self.loss_fn(
            y_hat, y, y_vol_hat, y_vol, mask, self.hier_params, self.global_params
        )

        # Log all loss components with 'train_' prefix
        for name, value in loss_components.items():
            self.log(f"train_{name}", value)
        return loss

    def validation_step(self, batch: TensorDataset) -> torch.Tensor:
        y, y_vol, mask, _, _, _, _ = batch
        y_hat, y_vol_hat = self(batch)
        loss, loss_components = self.loss_fn(
            y_hat, y, y_vol_hat, y_vol, mask, self.hier_params, self.global_params
        )

        # Log all loss components with 'val_' prefix
        for name, value in loss_components.items():
            self.log(f"val_{name}", value)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(self.parameters(), lr=0.01, weight_decay=0.0)
        return optimizer
