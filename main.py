import argparse

import lightning as L

from src.dataset import Dataset
from src.model import HierarchicalRegressionModel


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

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
