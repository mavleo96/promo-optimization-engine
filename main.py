import argparse

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset

from src.constants import EPS
from src.dataset import Dataset
from src.model import HierarchicalRegressionModel


class PredictionLogger(L.Callback):
    def __init__(self, dataset: TensorDataset):
        super().__init__()
        self.dataset = dataset

    def on_train_end(self, trainer: L.Trainer, pl_module: HierarchicalRegressionModel):
        # Get the full dataset and move to the correct device
        tensors = [t.to(pl_module.device) for t in self.dataset.tensors]
        y, y_vol, mask, time_index, nr_lag, macro, discount = tensors

        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            y_hat, y_vol_hat = pl_module(tensors)

            # Sum across SKU dimension (dim=1)
            y_hat_sum = y_hat.sum(dim=1)
            y_sum = y.sum(dim=1)
            y_vol_sum = y_vol.sum(dim=1)
            y_vol_hat_sum = y_vol_hat.sum(dim=1)

            # Move to CPU for plotting
            y_hat_sum = y_hat_sum.cpu()
            y_sum = y_sum.cpu()
            y_vol_sum = y_vol_sum.cpu()
            y_vol_hat_sum = y_vol_hat_sum.cpu()

        # Calculate metrics
        wmape = ((y_hat_sum - y_sum).abs() / (y_sum.abs() + EPS)).mean()
        wmape_vol = ((y_vol_hat_sum - y_vol_sum).abs() / (y_vol_sum.abs() + EPS)).mean()

        # Log model weights
        for name, param in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(
                f"weights/{name}",
                param.apply_activation().data.cpu(),
                global_step=trainer.global_step,
            )

        # Log overall metrics directly to logger
        trainer.logger.log_metrics(
            {
                "final_wmape": wmape.item(),
                "final_wmape_vol": wmape_vol.item(),
            },
            step=trainer.global_step,
        )

        # Create and save plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_sum.numpy(), label="actual", alpha=0.7)
        plt.plot(y_hat_sum.numpy(), label="predicted", alpha=0.7)
        plt.title("total net revenue: actual vs predicted")
        plt.xlabel("time")
        plt.ylabel("net revenue")
        plt.legend()
        trainer.logger.experiment.add_figure(
            "total_net_revenue_comparison", plt.gcf(), global_step=trainer.global_step
        )
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(y_vol_sum.numpy(), label="actual", alpha=0.7)
        plt.plot(y_vol_hat_sum.numpy(), label="predicted", alpha=0.7)
        plt.title("total volume: actual vs predicted")
        plt.xlabel("time")
        plt.ylabel("volume")
        plt.legend()

        trainer.logger.experiment.add_figure(
            "total_volume_comparison", plt.gcf(), global_step=trainer.global_step
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=1000)
    args = parser.parse_args()

    data = Dataset(args.data_path)
    train_loader, val_loader = data.train_test_split(
        num_workers=2, persistent_workers=True
    )

    model = HierarchicalRegressionModel(data)
    prediction_logger = PredictionLogger(data.tensor_dataset)

    logger = TensorBoardLogger(save_dir=".")
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        callbacks=[prediction_logger],
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
