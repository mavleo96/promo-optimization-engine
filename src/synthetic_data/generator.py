"""
This module is used to generate synthetic data
"""

import os
import random
import string
import json

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product
from typing import Dict, List, Union, Iterable

from .base_generator import BaseSyntheticData


class SyntheticData(BaseSyntheticData):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path)

    def generate_datasets(self) -> None:
        """Sets the synthetic data based on the configuration"""

        self["macro_data"] = self.create_macro_data()

        super().generate_datasets()

    @staticmethod
    def random_data_generator(
        shape: int,
        p1: float = 1,
        p2: float = 1,
        scale: float = 100,
        dist: str = "beta",
        sparsity: float = 0,
    ) -> np.ndarray:
        """Method to generate random data based on specified distribution
        Allows for sparsity to be added to the data"""

        if dist == "beta":
            array = np.random.beta(p1, p2, shape) * scale
        elif dist == "normal":
            array = np.random.normal(p1, p2, shape)

        if sparsity:
            array = np.where(np.random.binomial(1, sparsity, shape), 0, array)

        return array

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
                if 2020 <= year <= 2021:
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
        years = self.generator_dict["year"]
        months = self.generator_dict["month"]
        skus = self.generator_dict["sku"]
        brands = self.generator_dict["brand"]
        packs = self.generator_dict["pack"]
        sizes = self.generator_dict["size"]

        data = []

        for year in years:
            for month in months:
                for i, sku in enumerate(skus):
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
                    if 2020 <= year <= 2021:
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
                            brands[i],
                            packs[i],
                            sizes[i],
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

        df = pd.DataFrame(
            data,
            columns=[
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
            ],
        )
        return df
