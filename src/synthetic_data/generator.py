"""
This module is used to generate synthetic data
"""

# TODO: Make the data generation more realistic

import os
import random
import string
import json

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product
from typing import Dict, List, Union, Iterable
from .utils import (
    random_string_list,
    cross_join_data,
    random_map_join_data,
    random_data_generator,
)
from .base_generator import BaseSyntheticData


class SyntheticData(BaseSyntheticData):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)
        self.generator_dict["sku"] = random_string_list("sku", 1000)
        self.generator_dict["brand"] = random_string_list("brand", 20)
        self.generator_dict["pack"] = random_string_list("pack", 5)
        self.generator_dict["size"] = random_string_list("size", 3)
        self.generator_dict["price_segment"] = random_string_list("price_segment", 3)

    def generate_datasets(self) -> None:
        """Sets the synthetic data based on the configuration"""

        self["macro_data"] = self.create_macro_data()
        self["sales_data"] = self.create_sales_data()
        self["brand_segment_mapping"] = self.create_brand_segment_mapping()
        self["maximum_brand_discount_constraint"] = (
            self.create_brand_discount_constraint()
        )
        self["maximum_pack_discount_constraint"] = (
            self.create_pack_discount_constraint()
        )
        self["maximum_segment_discount_constraint"] = (
            self.create_segment_discount_constraint()
        )
        self["volume_variation_constraint"] = self.create_volume_variation_constraint()

        super().generate_datasets()

    def create_macro_data(self) -> pd.DataFrame:

        years = self.generator_dict["year"]
        months = self.generator_dict["month"]
        data = []

        base_retail_sales = 100  # Base index
        base_unemployment = 5.0  # Base %
        base_cpi = 100  # Base index
        base_private_consumption = 500  # Base consumption in billions
        base_gross_domestic_saving = 150  # Base savings in billions
        base_broad_money = 200  # Base money supply in billions
        base_gdp = 1000  # Base GDP in billions

        for year in years:
            for month in months:
                retail_sales = base_retail_sales * (1 + 0.01 * np.random.randn())
                unemployment_rate = base_unemployment + 0.1 * np.random.randn()
                cpi = base_cpi * (1 + 0.005 * np.random.randn())
                private_consumption = base_private_consumption * (
                    1 + 0.01 * np.random.randn()
                )
                gross_domestic_saving = base_gross_domestic_saving * (
                    1 + 0.015 * np.random.randn()
                )
                broad_money = base_broad_money * (1 + 0.012 * np.random.randn())
                gdp = base_gdp * (1 + 0.02 * np.random.randn())

                # Introduce COVID-19 effects (2020-2021)
                if (
                    self.time_config["covid_start"]["year"]
                    <= year
                    <= self.time_config["covid_end"]["year"]
                ) and (
                    self.time_config["covid_start"]["month"]
                    <= month
                    <= self.time_config["covid_end"]["month"]
                ):
                    # Adjust macroeconomic indicators
                    retail_sales *= np.random.uniform(0.7, 0.9)  # Drop in sales
                    unemployment_rate *= np.random.uniform(
                        1.2, 1.5
                    )  # Increase in unemployment
                    cpi *= np.random.uniform(
                        0.95, 1.05
                    )  # Slight inflation or deflation
                    private_consumption *= np.random.uniform(
                        0.7, 0.85
                    )  # Drop in spending
                    gross_domestic_saving *= np.random.uniform(
                        0.8, 1.1
                    )  # Mixed effects on savings
                    broad_money *= np.random.uniform(
                        1.1, 1.3
                    )  # Increase due to stimulus measures
                    gdp *= np.random.uniform(0.7, 0.85)  # Economic contraction

                data.append(
                    [
                        year,
                        month,
                        retail_sales,
                        unemployment_rate,
                        cpi,
                        private_consumption,
                        gross_domestic_saving,
                        broad_money,
                        gdp,
                    ]
                )

                # Introduce slight trend over time
                base_retail_sales *= 1.002
                base_unemployment *= 1 + np.random.uniform(-0.01, 0.01)
                base_cpi *= 1.002
                base_private_consumption *= 1.003
                base_gross_domestic_saving *= 1.002
                base_broad_money *= 1.004
                base_gdp *= 1.005

        df = pd.DataFrame(
            data,
            columns=[
                "year",
                "month",
                "retail_sales_index",
                "unemployment_rate",
                "cpi",
                "private_consumption",
                "gross_domestic_saving",
                "broad_money",
                "gdp",
            ],
        )
        return df

    def create_sales_data(self) -> pd.DataFrame:

        df = random_map_join_data(
            [
                self.generator_dict["sku"],
                self.generator_dict["brand"],
                self.generator_dict["pack"],
                self.generator_dict["size"],
            ],
        )

        data = []

        for year in self.generator_dict["year"]:
            for month in self.generator_dict["month"]:
                for sku in self.generator_dict["sku"]:
                    # Retrieve corresponding row with categorical values
                    row = df[df["sku"] == sku].iloc[0]

                    volume_hl = np.random.uniform(
                        10, 500
                    )  # Sales volume in hectoliters
                    gto = volume_hl * np.random.uniform(50, 150)  # Gross trade output
                    promotional_discount = gto * np.random.uniform(0.05, 0.2)
                    other_discount = gto * np.random.uniform(0.01, 0.05)
                    total_discount = promotional_discount + other_discount
                    excise = gto * np.random.uniform(0.02, 0.1)
                    net_revenue = gto - total_discount - excise
                    maco = net_revenue * np.random.uniform(
                        0.2, 0.4
                    )  # Margin contribution
                    vilc = net_revenue * np.random.uniform(
                        0.1, 0.3
                    )  # Variable indirect logistic cost

                    # Introduce COVID-19 effects (2020-2021)
                    if (
                        self.time_config["covid_start"]["year"]
                        <= year
                        <= self.time_config["covid_end"]["year"]
                    ) and (
                        self.time_config["covid_start"]["month"]
                        <= month
                        <= self.time_config["covid_end"]["month"]
                    ):
                        volume_hl *= np.random.uniform(0.6, 0.8)  # Drop in sales volume
                        gto *= np.random.uniform(
                            0.6, 0.8
                        )  # Lower revenue due to demand drop
                        promotional_discount *= np.random.uniform(
                            1.2, 1.5
                        )  # Increased promotions
                        other_discount *= np.random.uniform(1.1, 1.3)
                        total_discount = promotional_discount + other_discount
                        excise *= np.random.uniform(
                            0.9, 1.1
                        )  # Minor variation in tax impact
                        net_revenue = gto - total_discount - excise
                        maco *= np.random.uniform(0.5, 0.8)  # Lower profitability
                        vilc *= np.random.uniform(1.1, 1.3)  # Increased logistics costs

                    data.append(
                        [
                            year,
                            month,
                            sku,
                            row["brand"],
                            row["pack"],
                            row["size"],
                            volume_hl,
                            gto,
                            promotional_discount,
                            other_discount,
                            total_discount,
                            excise,
                            net_revenue,
                            maco,
                            vilc,
                        ]
                    )

        # Create a DataFrame from the generated data
        columns = [
            "year",
            "month",
            "sku",
            "brand",
            "pack",
            "size",
            "volume_hl",
            "gto",
            "promotional_discount",
            "other_discount",
            "total_discount",
            "excise",
            "net_revenue",
            "maco",
            "vilc",
        ]

        return pd.DataFrame(data, columns=columns)

    def create_brand_segment_mapping(self) -> pd.DataFrame:
        df = random_map_join_data(
            [self.generator_dict["brand"], self.generator_dict["price_segment"]],
        )
        return df

    def create_brand_discount_constraint(self) -> pd.DataFrame:

        df = pd.DataFrame(
            {
                "brand": self.generator_dict["brand"],
                "discount": np.random.uniform(
                    0.05, 0.3, len(self.generator_dict["brand"])
                ),
            }
        )
        return df

    def create_pack_discount_constraint(self) -> pd.DataFrame:

        df = pd.DataFrame(
            {
                "pack": self.generator_dict["pack"],
                "discount": np.random.uniform(
                    0.05, 0.3, len(self.generator_dict["pack"])
                ),
            }
        )
        return df

    def create_segment_discount_constraint(self) -> pd.DataFrame:

        df = pd.DataFrame(
            {
                "price_segment": self.generator_dict["price_segment"],
                "discount": np.random.uniform(
                    0.05, 0.3, len(self.generator_dict["price_segment"])
                ),
            }
        )
        return df

    def create_volume_variation_constraint(self) -> pd.DataFrame:

        df = self["sales_data"][["sku", "brand"]].drop_duplicates()
        df["min_volume_variation"] = 1 - random_data_generator(
            df.shape[0], 0, 0.2, dist="uniform"
        )
        df["max_volume_variation"] = 1 + random_data_generator(
            df.shape[0], 0, 0.2, dist="uniform"
        )

        return df
