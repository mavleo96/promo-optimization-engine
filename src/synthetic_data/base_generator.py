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
from .utils import (
    random_string_list,
    cross_join_data,
    random_map_join_data,
    random_data_generator,
)


class BaseSyntheticData(dict):

    def __init__(self, path: Union[str, Path]) -> None:

        path = Path(path)
        with open(path / "columns.json", "r") as c, open(path / "data.json", "r") as d:
            self.data_config = json.loads(d.read())
            self.columns = json.loads(c.read())
            if "year" in self.columns or "month" in self.columns:
                self.time_config = json.loads(open(path / "time.json", "r").read())
            else:
                self.time_config = {}

        self.generator_dict = self.column_data_generator_dict()
        super().__init__()

    def column_data_generator_dict(self) -> Dict[str, pd.Series]:

        generator_dict = {}
        for col, col_config in self.columns.items():
            if col == "year":
                generator_dict[col] = pd.Series(
                    np.arange(
                        self.time_config["start_year"],
                        self.time_config["end_year"] + 1,
                    ),
                    name=col,
                )
            elif col == "month":
                generator_dict[col] = pd.Series(np.arange(1, 13), name=col)
            elif col_config["type"] == "str":
                generator_dict[col] = random_string_list(
                    col, 2
                )  # TODO: Increase number of strings?
        return generator_dict

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
        df = cross_join_data(data_list)

        # Creating numerical columns
        for col in num_list:
            df[col] = random_data_generator(df.shape[0])
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
