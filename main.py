import argparse
import lightning as L
from src.dataset import Dataset
from src.model import HierarchicalRegressionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data = Dataset(args.data_path, device=args.device)
    train_loader, val_loader = data.train_test_split()

    model = HierarchicalRegressionModel(data)
    print(model)
    # TODO: check why parameters are not registered
    # LightningModule may have problems or there is a bug in the model class
    print(list(model.parameters()))

    trainer = L.Trainer(
        max_epochs=1000, accelerator="cpu", devices=1, log_every_n_steps=5
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
