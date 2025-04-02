import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.dataset import Dataset
from src.model_components import (BaselineLayer, DiscountLayer,
                                  MixedEffectLayer, VolumeConversion)


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

        self.hier_var_list = [
            *self.baseline_layer.hier_var_list,
            *self.me_layer.hier_var_list,
            *self.discount_layer.hier_var_list,
        ]
        self.loss_fn = nn.MSELoss()
        # self.l2_reg = nn.L1Loss()

    def forward(self, x):
        # TODO: assert x on shapes
        _, _, _, time_index, nr_lag, macro, discount = x

        # Use layers from ModuleList
        baseline = self.baseline_layer(time_index, nr_lag)
        mixed_effect, roi_mult = self.me_layer(macro)
        discount_uplift = self.discount_layer(discount)

        nr_pred = baseline * torch.prod(mixed_effect, dim=2) + torch.sum(
            discount_uplift, dim=2
        ) * torch.prod(roi_mult, dim=2)
        volume_pred = self.convert_to_volume(nr_pred)

        return nr_pred, volume_pred

    # Define the training loop
    def training_step(self, batch, batch_idx):

        y, y_vol, mask, time_index, nr_lag, macro, discount = batch
        y_hat, y_vol_hat = self(batch)
        mse_loss = self.loss_fn(y_hat, y) + self.loss_fn(y_vol_hat, y_vol)
        hier_reg = 0.01 * sum(torch.sum(i**2) for i in self.hier_var_list)
        loss = mse_loss + hier_reg
        self.log("mse_loss", mse_loss)
        self.log("hier_reg", hier_reg)
        self.log("train_loss", loss)  # Logging
        return loss

    # Validation loop
    def validation_step(self, batch, batch_idx):
        y, y_vol, mask, time_index, nr_lag, macro, discount = batch
        y_hat, y_vol_hat = self(batch)
        mse_loss = self.loss_fn(y_hat, y) + self.loss_fn(y_vol_hat, y_vol)
        hier_reg = 0  # 0.01 * sum(torch.sum(i**2) for i in self.hier_var_list)
        loss = mse_loss + hier_reg
        self.log("val_loss", loss)  # Logging
        return loss

    # Optimizer setup
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        return optimizer
