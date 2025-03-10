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
from typing import Dict, List, Union, Iterable


class BaseSyntheticData(dict):

    def __init__(self, path: Union[str, Path]) -> None:

        path = Path(path)
        with open(path / "columns.json", "r") as c, open(path / "data.json", "r") as d:
            self.data_config = json.loads(d.read())
            self.columns = json.loads(c.read())

        self.generator_dict = self.column_data_generator_dict()
        super().__init__()

    @staticmethod
    def random_string_list(name: str, n: int = 2) -> pd.Series:
        return pd.Series(
            ["".join(random.choices(string.ascii_letters, k=5)) for _ in range(n)],
            name=name,
        )

    @staticmethod
    def random_data_generator(
        shape: int, p1: float = 1, p2: float = 1, scale: float = 100
    ) -> np.ndarray:
        return np.random.beta(p1, p2, shape) * scale

    def column_data_generator_dict(self) -> Dict[str, pd.Series]:

        generator_dict = {}
        for col, col_config in self.columns.items():
            if col == "year":
                generator_dict[col] = pd.Series(np.arange(2018, 2024), name=col)
            elif col == "month":
                generator_dict[col] = pd.Series(np.arange(1, 13), name=col)
            elif col_config["type"] == "str":
                generator_dict[col] = self.random_string_list(
                    col, 2
                )  # TODO: Increase number of strings?
        return generator_dict

    @staticmethod
    def cross_join_data(dlist: List[Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
        """Method to cross join multiple dataframes/series"""

        a = dlist[0]
        if not isinstance(a, pd.DataFrame):
            a = a.to_frame()
        for b in dlist[1:]:
            a = a.merge(b, how="cross")
        return a

    @staticmethod
    def fake_map_join_data(dlist: List[Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
        """Method to join multiple dataframes/series based on the first element using dummy mapping"""

        return pd.concat(
            [
                dlist[0],  # First element is the base dataframe
                *[  # Remaining elements are randomly sampled
                    i.sample(n=len(dlist[0]), replace=True, ignore_index=True)
                    for i in dlist[1:]
                ],
            ],
            axis=1,
        )

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> List[str]:
        """Method to get the categorical columns of a dataframe"""
        return df.select_dtypes(exclude=["float", "int"]).columns.tolist()

    @staticmethod
    def get_numerical_columns(df: pd.DataFrame) -> List[str]:
        """Method to get the numerical columns of a dataframe"""
        return df.select_dtypes(include=["float", "int"]).columns.tolist()

    def create_fake_data(self, data_name: str) -> pd.DataFrame:
        """Creates and returns a fake dataframe based on the configuration"""

        column_list = self.data_config[data_name]

        cat_list = [
            i
            for i in column_list
            if (self.columns[i]["type"] == "str")
            or (self.columns[i]["group"] == "time")
        ]
        num_list = [
            i
            for i in column_list
            if not (
                (self.columns[i]["type"] == "str")
                or (self.columns[i]["group"] == "time")
            )
        ]

        # Creating categorical df and changing the frequency if required
        data_list = [self.generator_dict[i] for i in cat_list]
        df = self.cross_join_data(data_list)

        # Creating numerical columns
        for col in num_list:
            df[col] = self.random_data_generator(df.shape[0])
            if self.columns[col]["type"] == "int":
                df[col] = df[col].astype(int)

        return df

    def generate_datasets(self) -> None:
        for data_name in self.data_config:
            if data_name not in self:
                self[data_name] = self.create_fake_data(data_name)

    def export_datasets(self, path: Union[str, Path, None] = None) -> None:
        """Method to export the datasets to a csv file"""

        if path is None:
            path = Path("data")
            os.makedirs(path, exist_ok=True)
        else:
            path = Path(path)

        for data_name, data in self.items():
            data.to_csv(path / f"{data_name}.csv", index=False)
