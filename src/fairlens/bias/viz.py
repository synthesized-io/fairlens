from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from . import utils
from .exceptions import UnsupportedDistributionError


def plt_group_dist(df: pd.DataFrame, group1: Dict[str, List[Any]], group2: Dict[str, List[Any]], target_attr: str):
    df = utils.infer_dtype(df, target_attr)
    distr_type = utils.infer_distr_type(df[target_attr])

    if not (distr_type.is_continuous() or str(df[target_attr].dtype) in ["float64", "int64"]):
        raise UnsupportedDistributionError()

    pred1 = False
    for attr, vals in group1.items():
        for val in vals:
            pred1 |= df[attr] == val

    pred2 = False
    for attr, vals in group2.items():
        for val in vals:
            pred2 |= df[attr] == val

    plt.hist(df[pred1][target_attr], bins=100)
    plt.hist(df[pred2][target_attr], bins=100)
