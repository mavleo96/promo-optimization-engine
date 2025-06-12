import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset
from torchmetrics.functional import weighted_mean_absolute_percentage_error as wmape

from src.constants import EPS
from src.model import HierarchicalRegressionModel
from src.opt_engine import OptimizationEngine


class PredictionSaverCallback(L.Callback):
    def __init__(self, dataset: TensorDataset):
        super().__init__()
        self.dataset = dataset

    @torch.no_grad()
    def on_train_end(self, trainer: L.Trainer, pl_module: HierarchicalRegressionModel):
        data = [t.to(pl_module.device) for t in self.dataset.tensors]
        sales, volume, mask, _, _, _, _ = data

        pl_module.eval()
        sales_pred, volume_pred = pl_module(data)

        sales_pred[~mask] = 0
        volume_pred[~mask] = 0
        sales[~mask] = 0
        volume[~mask] = 0

        sales_pred_sum = sales_pred.sum(dim=1).cpu()
        sales_sum = sales.sum(dim=1).cpu()
        volume_sum = volume.sum(dim=1).cpu()
        volume_pred_sum = volume_pred.sum(dim=1).cpu()

        sales_wmape = wmape(sales_pred_sum, sales_sum)
        volume_wmape = wmape(volume_pred_sum, volume_sum)

        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_scalar(
                    "total_sales_wmape",
                    sales_wmape.item(),
                    global_step=trainer.global_step,
                )
                logger.experiment.add_scalar(
                    "total_volume_wmape",
                    volume_wmape.item(),
                    global_step=trainer.global_step,
                )

                plt.figure(figsize=(12, 6))
                plt.plot(sales_sum.numpy(), label="actual", alpha=0.7)
                plt.plot(sales_pred_sum.numpy(), label="predicted", alpha=0.7)
                plt.title("total sales: actual vs predicted")
                plt.xlabel("time")
                plt.ylabel("sales")
                plt.legend()

                logger.experiment.add_figure(
                    "total_sales_comparison", plt.gcf(), global_step=trainer.global_step
                )
                plt.close()

                plt.figure(figsize=(12, 6))
                plt.plot(volume_sum.numpy(), label="actual", alpha=0.7)
                plt.plot(volume_pred_sum.numpy(), label="predicted", alpha=0.7)
                plt.title("total volume: actual vs predicted")
                plt.xlabel("time")
                plt.ylabel("volume")
                plt.legend()

                logger.experiment.add_figure(
                    "total_volume_comparison",
                    plt.gcf(),
                    global_step=trainer.global_step,
                )
                plt.close()


class WeightSaverCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer: L.Trainer, pl_module: HierarchicalRegressionModel):
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                for name, param in pl_module.named_parameters():
                    logger.experiment.add_histogram(
                        f"weights/{name}",
                        param.data.cpu(),
                        global_step=trainer.global_step,
                    )


class OptimizationCallback(L.Callback):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_end(self, trainer: L.Trainer, pl_module: OptimizationEngine):
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                data = [t.to(pl_module.device) for t in pl_module.dataloader.dataset.tensors]  # type: ignore
                sales, _, mask, _, _, _, _ = data

                sales_pred, _ = pl_module.model(data)
                sales_pred[~mask] = 0
                sales[~mask] = 0
                sales_pred_sum = sales_pred.sum(dim=1).cpu()
                sales_sum = sales.sum(dim=1).cpu()
                sales_wmape = wmape(sales_pred_sum, sales_sum)

                opt_sales, _ = pl_module.model([*data[:-1], pl_module.optimized_spend])
                opt_sales[~mask] = 0
                opt_sales_sum = opt_sales.sum(dim=1).cpu()
                roi = (opt_sales_sum.sum() - sales_sum.sum()) / (
                    pl_module.optimized_spend.sum().item() + EPS  # type: ignore
                )

                score = (sales_wmape + roi) * 100

                logger.experiment.add_scalar(
                    "final_sales_wmape",
                    sales_wmape.item(),
                    global_step=trainer.global_step,
                )
                logger.experiment.add_scalar("score", score.item(), global_step=trainer.global_step)
