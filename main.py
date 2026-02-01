import argparse

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src.callback import (
    OptimizationCallback,
    PredictionSaverCallback,
    WeightSaverCallback,
)
from src.dataset import Dataset
from src.model import HierarchicalRegressionModel
from src.opt_engine import OptimizationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--opt_epochs", type=int, default=500)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    data = Dataset(args.data_path)
    train_loader, val_loader = data.train_val_dataloader(
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    model = HierarchicalRegressionModel(data)
    prediction_saver = PredictionSaverCallback(data.tensor_dataset)
    weight_saver = WeightSaverCallback()

    model_logger = TensorBoardLogger(save_dir=".", version=args.run_name, sub_dir="model")
    model_trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        callbacks=[prediction_saver, weight_saver],
        logger=model_logger,
    )
    model_trainer.fit(model, train_loader, val_loader)

    opt_engine = OptimizationEngine(model, data)
    opt_callback = OptimizationCallback()

    opt_logger = TensorBoardLogger(save_dir=".", version=args.run_name, sub_dir="opt")
    opt_trainer = L.Trainer(
        max_epochs=args.opt_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        callbacks=[opt_callback],
        logger=opt_logger,
    )
    opt_trainer.fit(opt_engine)


if __name__ == "__main__":
    main()
