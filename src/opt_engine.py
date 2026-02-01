import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import Dataset
from .loss import ROILoss
from .model import HierarchicalRegressionModel


class OptimizationEngine(L.LightningModule):
    def __init__(self, model: HierarchicalRegressionModel, dataset: Dataset):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False
        self.dataloader = dataset.test_dataloader()

        # Note: We directly optimize the raw spend tensor
        # However, it is better to optimize the softmax of raw spend tensor to avoid negative values
        self.register_parameter(
            "optimized_spend", nn.Parameter(self.dataloader.dataset.tensors[6].clone())
        )  # type: ignore

        self.loss_fn = ROILoss(
            dataset.constraint_tensors,
            dataset.gather_indices,
        )

    def forward(self) -> torch.Tensor:
        return self.optimized_spend  # type: ignore

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        init_sales, _ = self.model(batch)
        opt_sales, opt_vol = self.model([*batch[:-1], self.optimized_spend])

        loss, metrics = self.loss_fn(self.optimized_spend, opt_sales, init_sales, opt_vol)
        self.log_dict({f"opt/{k}": v for k, v in metrics.items()})
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        return self.dataloader
