import random
import string
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def random_data_generator(
    shape: int,
    p1: float = 1,
    p2: float = 1,
    scale: float = 100,
    dist: str = "beta",
    sparsity: float = 0,
    round: Union[int, None] = None,
) -> NDArray:
    """Method to generate random data based on specified distribution
    Allows for sparsity to be added to the data"""

    if dist == "beta":
        array = np.random.beta(p1, p2, shape) * scale
    elif dist == "normal":
        array = np.random.normal(p1, p2, shape)
    elif dist == "uniform":
        array = np.random.uniform(p1, p2, shape)

    if sparsity:
        array = np.where(np.random.binomial(1, sparsity, shape), 0, array)

    if round is not None:
        array = np.round(array, round)

    return array


def random_string_list(name: str, n: int = 2) -> pd.Series:
    """Method to generate a list of random strings"""
    return pd.Series(
        ["".join(random.choices(string.ascii_letters, k=5)) for _ in range(n)],
        name=name,
    ).sort_values(ignore_index=True)


def cross_join_data(dlist: Sequence[Union[pd.Series, pd.DataFrame]]) -> pd.DataFrame:
    """Method to cross join multiple dataframes/series"""

    a = dlist[0]
    if not isinstance(a, pd.DataFrame):
        a = a.to_frame()
    for b in dlist[1:]:
        a = a.merge(b, how="cross")
    return a


def random_map_join_data(
    dlist: Sequence[Union[pd.Series, pd.DataFrame]],
) -> pd.DataFrame:
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


def rolling_mean(var, window):
    """Method to calculate the rolling mean of an array"""
    width = [(window - 1, 0)] + [(0, 0) for _ in range(var.ndim - 1)]
    var = np.pad(var, width, mode="edge")
    var = np.lib.stride_tricks.sliding_window_view(var, window, axis=0)
    return var.mean(axis=-1)


def seasonality(freq, scale):
    """Method to calculate the seasonal component for a given frequency and scale"""
    scale = np.asarray(scale)
    base = np.sin(2 * np.pi * freq - np.pi / 2)
    if scale.ndim == 0:
        return 1 + base * scale
    else:
        base = base[:, np.newaxis]
        scale = scale[np.newaxis, :]
        return 1 + base * scale


def trend(time, scale):
    """Method to calculate the trend component for a given time and scale"""
    scale = np.asarray(scale)
    if scale.ndim == 0:
        return 1 + scale * time
    else:
        time = time[:, np.newaxis]
        scale = scale[np.newaxis, :]
        return 1 + scale * time


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Method to get the categorical columns of a dataframe"""
    return df.select_dtypes(exclude=["float", "int"]).columns.tolist()


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Method to get the numerical columns of a dataframe"""
    return df.select_dtypes(include=["float", "int"]).columns.tolist()
