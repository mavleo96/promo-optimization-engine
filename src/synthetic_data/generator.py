"""
This module is used to generate synthetic data that follows the model structure
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from .base_generator import BaseSyntheticData
from .utils import cross_join_data, random_data_generator, random_map_join_data, random_string_list


class SyntheticData(BaseSyntheticData):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)
        path = Path(path)

        with open(path / "macro_data.json", "r") as f:
            self.macro_data_config = json.load(f)

        self._initialize_generator_dict()
        self.params = self._initialize_params()

    def _initialize_generator_dict(self) -> None:
        """Initialize the dictionary of random string lists for different entities."""
        self.generator_dict["sku"] = random_string_list("sku", 1000)
        self.generator_dict["brand"] = random_string_list("brand", 20)
        self.generator_dict["pack"] = random_string_list("pack", 5)
        self.generator_dict["size"] = random_string_list("size", 3)
        self.generator_dict["price_segment"] = random_string_list("price_segment", 3)

    def generate_datasets(self) -> None:
        """Generate all synthetic datasets based on the configuration."""
        self["brand_segment_mapping"] = self.create_brand_segment_mapping()
        self["macro_data"] = self.create_macro_data()
        self["sales_data"] = self.create_sales_data()
        self["maximum_brand_discount_constraint"] = self.create_brand_discount_constraint()
        self["maximum_pack_discount_constraint"] = self.create_pack_discount_constraint()
        self["maximum_segment_discount_constraint"] = self.create_segment_discount_constraint()
        self["volume_variation_constraint"] = self.create_volume_variation_constraint()

        super().generate_datasets()

    def _initialize_params(self) -> Dict[str, Dict]:
        """Initialize all parameter dictionaries."""
        return {
            "brand": self._initialize_brand_params(),
            "segment": self._initialize_segment_params(),
            "pack": self._initialize_pack_params(),
            "size": self._initialize_size_params(),
            "macro": self._initialize_macro_params(),
        }

    def _initialize_segment_params(self) -> Dict:
        """Initialize price segment parameters."""
        return {
            segment: {"price_range": (100 + i * 50, 100 + (i + 1) * 50)}
            for i, segment in enumerate(self.generator_dict["price_segment"])
        }

    def _initialize_brand_params(self) -> Dict:
        """Initialize brand-specific parameters."""
        return {
            brand: {
                "price_elasticity": np.random.uniform(0.1, 0.9),
                "discount_sensitivity": np.random.uniform(0.5, 3),
                "macro_sensitivity": np.random.uniform(0.5, 1.5),
                "trend_rate": np.random.uniform(0.02, 0.05) / 12,
                "marketing_budget_ratio": np.random.uniform(0.02, 0.1),
            }
            for brand in self.generator_dict["brand"]
        }

    def _initialize_pack_params(self) -> Dict:
        """Initialize pack-specific parameters."""
        return {
            pack: {
                "price_multiplier": 0.003 * (3 * i if i else 1),
                "discount_multiplier": np.random.uniform(0.95, 1.05),
            }
            for i, pack in enumerate(self.generator_dict["pack"])
        }

    def _initialize_size_params(self) -> Dict:
        """Initialize size-specific parameters."""
        return {
            size: {
                "price_multiplier": 1.5 + 0.5 * i,
                "discount_multiplier": np.random.uniform(0.95, 1.05),
            }
            for i, size in enumerate(self.generator_dict["size"])
        }

    def _initialize_macro_params(self) -> Dict:
        """Initialize macroeconomic parameters."""
        return {
            i: {
                "trend_rate": np.random.uniform(-0.001, 0.002) / 12,
                "base": config["base_value"],
                "me_multiplier": np.random.uniform(-1, 1),
                "roi_multiplier": np.random.uniform(-0.3, 0.3),
            }
            for i, config in self.macro_data_config.items()
        }

    def create_brand_segment_mapping(self) -> pd.DataFrame:
        """Create mapping between brands and price segments."""
        df = random_map_join_data(
            [self.generator_dict["brand"], self.generator_dict["price_segment"]],
        )
        return df[self.data_config["brand_segment_mapping"]]

    def create_macro_data(self) -> pd.DataFrame:
        """Create macroeconomic data."""
        params = pd.DataFrame(self.params["macro"]).T.rename_axis("variable").reset_index()
        df = cross_join_data([self.time_data, params])

        df["trend"] = 1 + df.trend_rate * df.time_index
        df["variance"] = np.random.normal(loc=df.base, scale=df.base * 0.05, size=df.shape[0])
        df["variance"] = df.groupby("variable").variance.transform(
            lambda x: x.rolling(window=9, min_periods=1).mean()
        )
        df["value"] = df.variance * df.trend

        df = (
            df.groupby([*self.time_columns, "variable"])["value"]
            .sum()
            .unstack("variable")
            .reset_index()
            .rename_axis(None, axis=1)
        )
        return df[self.data_config["macro_data"]]

    def _calculate_mixed_effects(self) -> pd.DataFrame:
        """Calculate mixed effects from macroeconomic variables."""
        me_mult = {i: j["me_multiplier"] for i, j in self.params["macro"].items()}
        roi_mult = {i: j["roi_multiplier"] for i, j in self.params["macro"].items()}

        var_names = [i for i in self.data_config["macro_data"] if i not in self.time_columns]
        df = self["macro_data"].copy()

        normalize = lambda x: (x / x.mean() - 1)
        pertubate_func = lambda x, mult: 1 + np.tanh(x * mult[x.name])

        df[var_names] = df[var_names].apply(normalize, axis=0)

        df["me_effect"] = df[var_names].apply(pertubate_func, mult=me_mult).prod(axis=1)
        df["roi_mult"] = df[var_names].apply(pertubate_func, mult=roi_mult).prod(axis=1)

        return df[[*self.time_columns, "me_effect", "roi_mult"]]

    def create_sales_data(self) -> pd.DataFrame:
        """Create sales data with all related metrics."""
        macro_effect = self._calculate_mixed_effects()
        sku_list = self._create_sku_list()
        params = self._combine_params(sku_list)

        df = cross_join_data([self.time_data, params])
        df = df.merge(macro_effect, how="left")

        df = self._calculate_base_metrics(df)
        df = self._calculate_discount_metrics(df)
        df = self._calculate_final_metrics(df)

        return df[self.data_config["sales_data"]]

    def _create_sku_list(self) -> pd.DataFrame:
        """Create the base SKU list with all attributes."""
        sku_list = random_map_join_data(
            [
                self.generator_dict["sku"],
                self.generator_dict["brand"],
                self.generator_dict["pack"],
                self.generator_dict["size"],
            ],
        )
        return sku_list.merge(self["brand_segment_mapping"])

    def _calculate_base_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate base metrics for sales data."""
        df["price"] = df.base_price * random_data_generator(df.shape[0], 1, 0.05, 1, dist="normal")

        df["trend"] = 1 + df.trend_rate * df.time_index
        df["organic_sales"] = (
            df.base_value
            * df.trend
            * (1 + df.macro_sensitivity * (df.me_effect - 1))
            * (df.price / df.base_price) ** -df.price_elasticity
        )
        return df

    def _calculate_discount_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate discount-related metrics."""
        df["promotional_discount"] = (
            0.5
            * df.marketing_budget_ratio
            * df.base_value
            * random_data_generator(df.shape[0], p1=2, p2=5, scale=1, dist="beta", sparsity=0.3)
        )
        df["other_discount"] = (
            0.5
            * df.marketing_budget_ratio
            * df.base_value
            * random_data_generator(df.shape[0], p1=2, p2=5, scale=1, dist="beta", sparsity=0.7)
        )
        df["total_discount"] = df.promotional_discount + df.other_discount
        df["discount_uplift"] = (
            df.discount_sensitivity
            * df.total_discount
            * (1 + df.macro_sensitivity * (df.roi_mult - 1))
        )
        return df

    def _calculate_final_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final sales metrics."""
        df["gto"] = df.organic_sales + df.discount_uplift + np.random.normal(0.05 * df.base_value)
        df["excise"] = df.gto * random_data_generator(df.shape[0], 0.12, 0.15, 1, dist="uniform")
        df["vilc"] = df.gto * random_data_generator(df.shape[0], 0.09, 0.11, 1, dist="uniform")

        df["volume_hl"] = df.gto / df.price
        df["net_revenue"] = df.gto - df.total_discount - df.excise
        df["maco"] = df.net_revenue - df.vilc
        return df

    def _combine_params(self, sku_list: pd.DataFrame) -> pd.DataFrame:
        """Combine all parameter dataframes into a single dataframe."""
        brand_params = pd.DataFrame(self.params["brand"]).T.rename_axis("brand").reset_index()
        segment_params = (
            pd.DataFrame(self.params["segment"]).T.rename_axis("price_segment").reset_index()
        )
        pack_params = pd.DataFrame(self.params["pack"]).T.rename_axis("pack").reset_index()
        size_params = pd.DataFrame(self.params["size"]).T.rename_axis("size").reset_index()

        pack_size_params = pack_params.merge(size_params, how="cross", suffixes=("_pack", "_size"))

        params = (
            sku_list.merge(brand_params, how="left")
            .merge(segment_params, how="left")
            .merge(pack_size_params, how="left")
        )

        params["base_value"] = random_data_generator(params.shape[0], 2, 5, 100000, dist="beta")
        params["base_price"] = params.apply(lambda x: np.random.uniform(*x.price_range), axis=1)
        params["base_price"] *= params.price_multiplier_pack * params.price_multiplier_size
        params["discount_sensitivity"] *= (
            params.discount_multiplier_pack * params.discount_multiplier_size
        )
        return params[
            [
                *sku_list.columns,
                "base_value",
                "base_price",
                *[i for i in brand_params.columns if i != "brand"],
            ]
        ]

    def create_brand_discount_constraint(self) -> pd.DataFrame:
        """Create brand-specific discount constraints"""

        df = self.generator_dict["brand"].to_frame()

        discount_data = self["sales_data"][[*self.time_columns, "brand", "total_discount"]]
        discount_data = discount_data.groupby([*self.time_columns, "brand"]).total_discount.sum()
        discount_data = discount_data.groupby("brand").mean().values

        df["discount"] = discount_data * random_data_generator(
            df.shape[0], 1.1, 0.1, 1, dist="normal"
        )
        return df

    def create_pack_discount_constraint(self) -> pd.DataFrame:
        """Create pack-specific discount constraints"""
        df = self.generator_dict["pack"].to_frame()

        discount_data = self["sales_data"][[*self.time_columns, "pack", "total_discount"]]
        discount_data = discount_data.groupby([*self.time_columns, "pack"]).total_discount.sum()
        discount_data = discount_data.groupby("pack").mean().values

        df["discount"] = discount_data * random_data_generator(
            df.shape[0], 1.1, 0.1, 1, dist="normal"
        )
        return df

    def create_segment_discount_constraint(self) -> pd.DataFrame:
        """Create segment-specific discount constraints."""
        df = self.generator_dict["price_segment"].to_frame()

        discount_data = self["sales_data"].merge(self["brand_segment_mapping"], how="left")
        discount_data = discount_data[[*self.time_columns, "price_segment", "total_discount"]]
        discount_data = discount_data.groupby(
            [*self.time_columns, "price_segment"]
        ).total_discount.sum()
        discount_data = discount_data.groupby("price_segment").mean().values

        df["discount"] = discount_data * random_data_generator(
            df.shape[0], 1.1, 0.1, 1, dist="normal"
        )
        return df

    def create_volume_variation_constraint(self) -> pd.DataFrame:
        """Create volume variation constraints."""
        df = self["sales_data"][["sku", "brand"]].drop_duplicates()

        df["min_volume_variation"] = random_data_generator(df.shape[0], 0.7, 1, 1, dist="uniform")
        df["max_volume_variation"] = random_data_generator(df.shape[0], 1, 1.3, 1, dist="uniform")
        return df
