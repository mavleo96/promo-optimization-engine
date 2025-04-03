from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.load()

        self.encodings = self.create_encodings()

        self.n_sku = len(self.encodings["label_dict"]["sku"])
        self.n_macro = len(self.encodings["label_dict"]["macro"])
        self.n_discount_type = len(self.encodings["label_dict"]["discount"])

        self.sales_index = self.encodings["sales_index"]
        self.sku_list = self.encodings["label_dict"]["sku"]

        self.filter_data()
        self.process_sales_data()
        self.process_macro_data()
        self.normalize_data()

        self.tensor_dataset = self.create_tensors()

    def load(self) -> None:
        self.raw_sales_data = self.read_csv("sales_data")
        self.raw_segment_mapping = self.read_csv("brand_segment_mapping")

        self.raw_macro_data = self.read_csv("macro_data")

        self.raw_brand_constraint = self.read_csv("maximum_brand_discount_constraint")
        self.raw_pack_constraint = self.read_csv("maximum_pack_discount_constraint")
        self.raw_segment_constraint = self.read_csv(
            "maximum_segment_discount_constraint"
        )
        self.raw_volume_constraint = self.read_csv("volume_variation_constraint")

    def read_csv(self, filename: str) -> pd.DataFrame:
        data = pd.read_csv(self.path / f"{filename}.csv")
        if "year" in data.columns and "month" in data.columns:
            data["date"] = data.apply(
                lambda x: datetime(int(x["year"]), int(x["month"]), 1), axis=1
            )
            data.drop(columns=["year", "month"], inplace=True)
        return data

    def create_encodings(self) -> Dict:
        master_mapping = (
            self.raw_sales_data[["sku", "brand", "pack", "size"]]
            .drop_duplicates()
            .merge(self.raw_segment_mapping)
        )

        def label_encoder(series):
            unique_values = series.sort_values().unique()
            return dict(zip(unique_values, range(len(unique_values))))

        def mapper(col_val, col_key="sku"):
            nonlocal master_mapping, label_dict
            df = master_mapping[[col_key, col_val]].drop_duplicates()
            df.loc[:, col_val] = df[col_val].map(label_dict[col_val])
            df.loc[:, col_key] = df[col_key].map(label_dict[col_key])

            return df.set_index(col_key).to_dict()[col_val]

        label_dict = {
            col: label_encoder(master_mapping[col]) for col in master_mapping.columns
        }
        mapper_dict = {
            col: mapper(col) for col in master_mapping.columns if col != "sku"
        }
        sales_index = pd.date_range(
            self.raw_sales_data.date.min(), self.raw_sales_data.date.max(), freq="MS"
        )

        label_dict["discount"] = {"other_discount": 0, "promotional_discount": 1}
        label_dict["macro"] = {
            "cpi": 0,
            "gross_domestic_saving": 1,
            "retail_sales_index": 2,
            "unemployment_rate": 3,
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

    def process_sales_data(self) -> None:
        sales_data = (
            self.raw_sales_data.groupby(["date", "sku"])
            .sum(numeric_only=True)
            .sort_index()
            .unstack("sku")
            .reindex(self.encodings["sales_index"])
            .fillna(0)
        )
        self.nr_data = sales_data["net_revenue"].clip(lower=0).fillna(0.0)
        self.nr_lag_data = (
            sales_data["net_revenue"]
            .shift(1)
            .clip(lower=0)
            .bfill()
            .mean(axis=1)
            # TODO: resolve overloading
        )
        self.mask = (self.nr_data >= 0.0).values

        self.volume_data = sales_data["volume_hl"].clip(lower=0).fillna(0)
        self.discount_data = (
            sales_data[self.encodings["label_dict"]["discount"].keys()]
            .swaplevel(0, 1, axis=1)
            .sort_index(axis=1)
        )
        self.base_init = self.nr_data.mean(axis=0)
        self.time_index = np.arange(len(self.sales_index)).reshape(-1, 1)

    def process_macro_data(self) -> None:
        self.macro_data = self.raw_macro_data.copy(deep=True)
        self.macro_data = self.macro_data.set_index("date").reindex(self.sales_index)
        self.macro_data = self.macro_data.interpolate(
            method="linear", limit_direction="both"
        )

        # Feature selected from the macro data based on correlation with sales and multicollinearity
        self.macro_data = self.macro_data.loc[
            :, self.encodings["label_dict"]["macro"].keys()
        ]
        # TODO: add covid flags here

    def normalize_data(self) -> None:
        self.scaler = self.nr_data.mean(axis=None)
        self.vol_scaler = self.volume_data.mean(axis=None)
        self.macro_scaler = self.macro_data.mean()

        self.nr = (self.nr_data / self.scaler).values
        self.nr_lag = (self.nr_lag_data / self.scaler).values
        self.nr_lag = np.expand_dims(self.nr_lag, 1)
        self.base_init = (self.base_init / self.scaler).values
        self.base_init = np.expand_dims(self.base_init, 0)

        self.volume = (self.volume_data / self.vol_scaler).values
        self.macro = (self.macro_data / self.macro_scaler - 1).values
        self.macro = np.expand_dims(self.macro, 1)

        self.discount = (self.discount_data / self.scaler).values
        self.discount = self.discount.reshape(
            len(self.sales_index), len(self.sku_list), -1
        )

    def create_tensors(self) -> TensorDataset:
        # Create a tensor dataset
        dataset = TensorDataset(
            torch.tensor(self.nr, dtype=torch.float32),
            torch.tensor(self.volume, dtype=torch.float32),
            torch.tensor(self.mask, dtype=torch.bool),
            torch.tensor(self.time_index, dtype=torch.float32),
            torch.tensor(self.nr_lag, dtype=torch.float32),
            torch.tensor(self.macro, dtype=torch.float32),
            torch.tensor(self.discount, dtype=torch.float32),
        )
        return dataset

    def train_test_split(
        self, train_size: float = 0.75, batch_size: int = 1024, *args, **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        # 3 fold temporal cross validation needed?
        # create dataloaders
        train_size = int(train_size * len(self.tensor_dataset))

        train_dataset = TensorDataset(*self.tensor_dataset[:train_size])
        val_dataset = TensorDataset(*self.tensor_dataset[train_size:])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, *args, **kwargs
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, *args, **kwargs
        )

        return train_loader, val_loader
