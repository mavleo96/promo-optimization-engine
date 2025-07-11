from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.path}")

        self.load()
        self.encodings = self.create_encodings()

        self.n_sku = len(self.encodings["label_dict"]["sku"])
        self.n_macro = len(self.encodings["label_dict"]["macro"])
        self.n_discount_type = len(self.encodings["label_dict"]["discount"])
        self.n_time = len(self.encodings["sales_index"])

        self.sales_index = self.encodings["sales_index"]
        self.sku_list = list(self.encodings["label_dict"]["sku"].keys())

        self.filter_data()
        self.process_sales_data()
        self.process_macro_data()
        self.process_constraint_data()
        self.normalize_data()

        self.tensor_dataset = self.create_tensors()
        self.gather_indices = self.create_gather_indices()
        self.constraint_tensors = self.create_constraint_tensors()

    def load(self) -> None:
        required_files = [
            "sales_data",
            "brand_segment_mapping",
            "macro_data",
            "maximum_brand_discount_constraint",
            "maximum_pack_discount_constraint",
            "maximum_segment_discount_constraint",
            "volume_variation_constraint",
        ]

        for file in required_files:
            if not (self.path / f"{file}.csv").exists():
                raise FileNotFoundError(f"Required file not found: {file}.csv")

        self.raw_sales_data = self.read_csv("sales_data")
        self.raw_segment_mapping = self.read_csv("brand_segment_mapping")
        self.raw_macro_data = self.read_csv("macro_data")
        self.raw_brand_constraint_data = self.read_csv("maximum_brand_discount_constraint")
        self.raw_pack_constraint_data = self.read_csv("maximum_pack_discount_constraint")
        self.raw_segment_constraint_data = self.read_csv("maximum_segment_discount_constraint")
        self.raw_volume_constraint_data = self.read_csv("volume_variation_constraint")

    def read_csv(self, filename: str) -> pd.DataFrame:
        data = pd.read_csv(self.path / f"{filename}.csv")
        if "year" in data.columns and "month" in data.columns:
            data["date"] = pd.to_datetime(data[["year", "month"]].assign(day=1))
            data.drop(columns=["year", "month"], inplace=True)
        return data

    def create_encodings(self) -> Dict[str, Any]:
        master_mapping = (
            self.raw_sales_data[["sku", "brand", "pack", "size"]]
            .drop_duplicates()
            .merge(self.raw_segment_mapping)
        )

        def label_encoder(series: pd.Series) -> Dict[str, int]:
            unique_values = series.sort_values().unique()
            return dict(zip(unique_values, range(len(unique_values))))

        def mapper(col_val: str, col_key: str = "sku") -> Dict[str, int]:
            nonlocal master_mapping, label_dict
            df = master_mapping[[col_key, col_val]].drop_duplicates()
            df.loc[:, col_val] = df[col_val].map(label_dict[col_val])
            df.loc[:, col_key] = df[col_key].map(label_dict[col_key])
            return df.set_index(col_key).to_dict()[col_val]

        label_dict = {col: label_encoder(master_mapping[col]) for col in master_mapping.columns}
        mapper_dict = {col: mapper(col) for col in master_mapping.columns if col != "sku"}
        sales_index = pd.date_range(
            self.raw_sales_data.date.min(), self.raw_sales_data.date.max(), freq="MS"
        )

        label_dict["discount"] = {"other_discount": 0, "promotional_discount": 1}
        label_dict["macro"] = {
            "cpi": 0,
            "gross_domestic_saving": 1,
            "retail_sales_index": 2,
            "unemployment_rate": 3,
            "covid": 4,
        }

        return {
            "label_dict": label_dict,
            "mapper_dict": mapper_dict,
            "sales_index": sales_index,
        }

    def filter_data(self) -> None:
        self.raw_sales_data = self.raw_sales_data[
            self.raw_sales_data["sku"].isin(self.sku_list)
            & self.raw_sales_data["date"].isin(self.sales_index)
        ]
        self.raw_macro_data = self.raw_macro_data[
            self.raw_macro_data["date"].isin(self.sales_index)
        ]
        self.raw_brand_constraint_data = self.raw_brand_constraint_data[
            self.raw_brand_constraint_data["brand"].isin(
                list(self.encodings["label_dict"]["brand"].keys())
            )
        ]
        self.raw_pack_constraint_data = self.raw_pack_constraint_data[
            self.raw_pack_constraint_data["pack"].isin(
                list(self.encodings["label_dict"]["pack"].keys())
            )
        ]
        self.raw_segment_constraint_data = self.raw_segment_constraint_data[
            self.raw_segment_constraint_data["price_segment"].isin(
                list(self.encodings["label_dict"]["price_segment"].keys())
            )
        ]
        self.raw_volume_constraint_data = self.raw_volume_constraint_data[
            self.raw_volume_constraint_data["sku"].isin(self.sku_list)
        ]

    def process_sales_data(self) -> None:
        sales_data = (
            self.raw_sales_data.groupby(["date", "sku"])
            .sum(numeric_only=True)
            .sort_index()
            .unstack("sku")
            .reindex(self.encodings["sales_index"])
            .fillna(0)
        )
        self.sales_data = sales_data["gto"].clip(lower=0).fillna(0.0)
        self.sales_lag_data = (
            sales_data["gto"].shift(1).clip(lower=0).bfill().mean(axis=1)  # type: ignore
        )
        self.mask = (self.sales_data >= 0.0).values

        self.volume_data = sales_data["volume_hl"].clip(lower=0).fillna(0)
        self.discount_data = (
            sales_data[list(self.encodings["label_dict"]["discount"].keys())]
            .swaplevel(0, 1, axis=1)  # type: ignore
            .sort_index(axis=1)
        )
        self.base_init = self.sales_data.mean(axis=0)
        self.time_index = np.arange(len(self.sales_index)).reshape(-1, 1)

    def process_macro_data(self) -> None:
        self.macro_data = self.raw_macro_data.copy(deep=True)
        self.macro_data = self.macro_data.set_index("date").reindex(self.sales_index)
        self.macro_data = self.macro_data.interpolate(method="linear", limit_direction="both")

        covid_start, covid_end = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-31")
        self.macro_data["covid"] = (self.sales_index >= covid_start) & (
            self.sales_index <= covid_end
        )

        # Feature selected from the macro data based on correlation with sales and multicollinearity
        self.macro_data = self.macro_data.loc[:, self.encodings["label_dict"]["macro"].keys()]

    def process_constraint_data(self) -> None:
        self.brand_constraint_data = self.raw_brand_constraint_data.set_index("brand").sort_index()
        self.pack_constraint_data = self.raw_pack_constraint_data.set_index("pack").sort_index()
        self.segment_constraint_data = self.raw_segment_constraint_data.set_index(
            "price_segment"
        ).sort_index()
        self.volume_constraint_data = (
            self.raw_volume_constraint_data.drop("brand", axis=1).set_index("sku").sort_index()
        )

    def normalize_data(self) -> None:
        self.scaler = self.sales_data.mean(axis=None)
        self.vol_scaler = self.volume_data.mean(axis=None)
        self.macro_scaler = self.macro_data.mean()
        self.macro_scaler["covid"] = 1.0

        self.sales = (self.sales_data / self.scaler).values
        self.sales_lag = np.array(self.sales_lag_data / self.scaler)
        self.sales_lag = np.expand_dims(self.sales_lag, 1)
        self.base_init = np.array(self.base_init / self.scaler)
        self.base_init = np.expand_dims(self.base_init, 0)

        self.volume = (self.volume_data / self.vol_scaler).values
        self.macro = (self.macro_data / self.macro_scaler - 1).values
        self.macro = np.expand_dims(self.macro, 1)

        self.discount = np.array(self.discount_data / self.scaler)
        self.discount = self.discount.reshape(len(self.sales_index), len(self.sku_list), -1)

        self.brand_constraint = (self.brand_constraint_data / self.scaler).T.values
        self.pack_constraint = (self.pack_constraint_data / self.scaler).T.values
        self.segment_constraint = (self.segment_constraint_data / self.scaler).T.values
        self.volume_constraint = self.volume_constraint_data.T.values

    def create_tensors(self) -> TensorDataset:
        return TensorDataset(
            torch.tensor(self.sales, dtype=torch.float32),
            torch.tensor(self.volume, dtype=torch.float32),
            torch.tensor(self.mask, dtype=torch.bool),
            torch.tensor(self.time_index, dtype=torch.float32),
            torch.tensor(self.sales_lag, dtype=torch.float32),
            torch.tensor(self.macro, dtype=torch.float32),
            torch.tensor(self.discount, dtype=torch.float32),
        )

    def create_gather_indices(self) -> Dict[str, torch.Tensor]:
        return {
            "brand": torch.tensor(
                [self.encodings["mapper_dict"]["brand"][i] for i in range(self.n_sku)],
                dtype=torch.long,
            ),
            "pack": torch.tensor(
                [self.encodings["mapper_dict"]["pack"][i] for i in range(self.n_sku)],
                dtype=torch.long,
            ),
            "price_segment": torch.tensor(
                [self.encodings["mapper_dict"]["price_segment"][i] for i in range(self.n_sku)],
                dtype=torch.long,
            ),
        }

    def create_constraint_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "brand": torch.tensor(self.brand_constraint, dtype=torch.float32),
            "pack": torch.tensor(self.pack_constraint, dtype=torch.float32),
            "price_segment": torch.tensor(self.segment_constraint, dtype=torch.float32),
            "volume_variation": torch.tensor(self.volume_constraint, dtype=torch.float32),
        }

    def train_val_dataloader(self, *args, **kwargs) -> Tuple[DataLoader, DataLoader]:
        train_size = int(self.n_time - 12)
        val_size = 9

        train_dataset = TensorDataset(*self.tensor_dataset[:train_size])
        val_dataset = TensorDataset(*self.tensor_dataset[train_size : train_size + val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=self.n_time, shuffle=True, *args, **kwargs
        )
        val_loader = DataLoader(val_dataset, batch_size=self.n_time, shuffle=False, *args, **kwargs)

        return train_loader, val_loader

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        test_size = 3
        test_dataset = TensorDataset(*self.tensor_dataset[-test_size:])
        test_loader = DataLoader(
            test_dataset, batch_size=self.n_time, shuffle=False, *args, **kwargs
        )
        return test_loader
