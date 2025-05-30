from typing import List, Tuple

import lightning as L
import torch
from torch.optim import AdamW

from .dataset import Dataset
from .loss import HierarchicalRegularizationLoss, RegressionLoss
from .model_components import BaselineLayer, DiscountLayer, MixedEffectLayer, VolumeConversion


class HierarchicalRegressionModel(L.LightningModule):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sku = dataset.n_sku
        self.n_macro = dataset.n_macro
        self.n_discount_type = dataset.n_discount_type
        self.baseline_init = dataset.base_init

        self.baseline_layer = BaselineLayer(hier_shape=self.n_sku, baseline_init=self.baseline_init)  # type: ignore
        self.me_layer = MixedEffectLayer(hier_shape=self.n_sku, n_macro=self.n_macro)
        self.discount_layer = DiscountLayer(hier_shape=self.n_sku, n_types=self.n_discount_type)
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
        self.hier_loss_fn = HierarchicalRegularizationLoss()
        self.reg_loss_fn = RegressionLoss()

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, time_index, sales_lag, macro, discount = x

        baseline = self.baseline_layer(time_index, sales_lag)
        mixed_effect, roi_mult = self.me_layer(macro)
        discount_uplift = self.discount_layer(discount)

        sales_pred = baseline * mixed_effect.prod(2) + discount_uplift.sum(2) * roi_mult.prod(2)
        volume_pred = self.convert_to_volume(sales_pred)

        return sales_pred, volume_pred

    def training_step(self, batch: List[torch.Tensor]) -> torch.Tensor:
        y, y_vol, mask, _, _, _, _ = batch
        y_hat, y_vol_hat = self(batch)
        reg_loss, reg_loss_components = self.reg_loss_fn(y_hat, y, y_vol_hat, y_vol, mask)
        hier_loss, hier_loss_components = self.hier_loss_fn(self.hier_params, self.global_params)
        loss = reg_loss + hier_loss

        self.log_dict({"train_" + k: v for k, v in reg_loss_components.items()})
        self.log_dict(hier_loss_components)
        self.log("train_total_loss", loss)
        return loss

    def validation_step(self, batch: List[torch.Tensor]) -> torch.Tensor:
        y, y_vol, mask, _, _, _, _ = batch
        y_hat, y_vol_hat = self(batch)
        reg_loss, reg_loss_components = self.reg_loss_fn(y_hat, y, y_vol_hat, y_vol, mask)

        self.log_dict({"val_" + k: v for k, v in reg_loss_components.items()})
        return reg_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(self.parameters(), lr=0.01)
        return optimizer
