from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .exceptions import InvalidAttributeError


class DistrType(Enum):
    """Indicates the type distribution of data in a series."""

    Continuous = "Continuous"
    Binary = "Binary"
    Categorical = "Categorical"

    def is_continuous(self):
        return self == DistrType.Continuous

    def is_binary(self):
        return self == DistrType.Binary

    def is_categorical(self):
        return self == DistrType.Categorical


def bin(
    df: pd.DataFrame,
    column_name: str,
    n_bins: Optional[int] = None,
    remove_outliers: Optional[float] = 0.1,
    quantile_based: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Bin continous values into discrete bins.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        column_name (str):
            The name of the dataframe column to transform.
        n_bins (Optional[int], optional):
            The number of bins. Defaults to Freedman-Diaconis rule.
        remove_outliers (Optional[float], optional):
            Any data point outside this quantile (two-sided) will be dropped before computing bins.
            If `None`, outliers are not removed. Defaults to 0.1.
        quantile_based (bool, optional):
            Whether the bin computation is quantile based. Defaults to False.

    Returns:
        pd.DataFrame:
            The binned dataframe.
    """

    column = df[column_name].copy()
    column_clean = column.dropna()

    n_bins = n_bins or fd_opt_bins(column)

    if remove_outliers:
        percentiles = [remove_outliers * 100.0 / 2, 100 - remove_outliers * 100.0 / 2]
        start, end = np.percentile(column_clean, percentiles)

        if start == end:
            start, end = min(column_clean), max(column_clean)

        column_clean = column_clean[(start <= column_clean) & (column_clean <= end)]

    if not quantile_based:
        _, bins = pd.cut(column_clean, n_bins, retbins=True, **kwargs)
    else:
        _, bins = pd.qcut(column_clean, n_bins, retbins=True, **kwargs)
    bins = list(bins)  # Otherwise it is np.ndarray
    bins[0], bins[-1] = column.min(), column.max()

    # Manually construct interval index for dates as pandas can't do a quantile date interval by itself.
    if isinstance(bins[0], pd.Timestamp):
        bins = pd.IntervalIndex([pd.Interval(bins[n], bins[n + 1]) for n in range(len(bins) - 1)], closed="left")

    df.loc[:, column_name] = pd.cut(df.loc[:, column_name], bins=bins, include_lowest=True, **kwargs)

    return df


def infer_dtype(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Infers the type of the data and converts the data to it.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        column_name (str):
            The name of the dataframe column to transform.

    Returns:
        pd.DataFrame:
            The dataframe converted to its inferred type.
    """

    column = df[column_name].copy()

    in_dtype = str(column.dtype)

    # Try to convert it to numeric
    if column.dtype.kind not in ("i", "u", "f"):
        n_nans = column.isna().sum()
        col_num = pd.to_numeric(column, errors="coerce")
        if col_num.isna().sum() == n_nans:
            column = col_num

    # Try to convert it to date
    if column.dtype.kind == "O":
        n_nans = column.isna().sum()
        try:
            col_date = pd.to_datetime(column, errors="coerce")
        except TypeError:  # Argument 'date_string' has incorrect type (expected str, got numpy.str_)
            col_date = pd.to_datetime(column.astype(str), errors="coerce")

        if col_date.isna().sum() == n_nans:
            column = col_date

    out_dtype = str(column.dtype)

    if out_dtype == in_dtype:
        return df
    elif out_dtype in ("i", "u", "f", "f8", "i8", "u8"):
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    else:
        df[column_name] = df[column_name].astype(out_dtype, errors="ignore")

    return df


def infer_distr_type(column: pd.Series, ctl_mult: float = 2.5, min_num_unique: int = 10) -> DistrType:
    """Infers whether the data in a column or series is continuous, categorical or binary.

    Args:
        column (pd.Series):
            The column from the data or data series to consider.
        ctl_mult (float, optional):
            Categorical threshold log multiplier. Defaults to 2.5.
        min_num_unique (int, optional):
            Minimum number of unique values for the data to be continuous. Defaults to 10.

    Returns:
        DistrType:
            The output is an enum representing the type of distribution.

    Examples:
        >>> col_type = infer_distr_type(range(1000))
        >>> col_type.is_continuous()
        True
        >>> col_type.is_binary()
        False
    """

    n_unique = column.nunique()
    num_rows = len(column)

    if n_unique > max(min_num_unique, ctl_mult * np.log(num_rows)):
        return DistrType.Continuous

    elif n_unique == 2:
        return DistrType.Binary

    else:
        return DistrType.Categorical


def get_predicates(
    df: pd.DataFrame, group1: Dict[str, List[str]], group2: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forms and returns pandas dataframe predicates which can be used to index group1 and group2.
    If group2 is None the second predicate returned is the inverse of that used to index group1.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        group1 (Dict[str, List[Any]]):
            The first group of interest.
        group2 (Optional[Dict[str, List[str]]], optional):
            The second group of interest. Defaults to None.

    Raises:
        InvalidAttributeError:
            Indicates an ill-formed group input due to invalid attributes in this case.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            A pair of dataframes that can be used to index the corresponding groups of data.
    """

    # Check all attributes are valid
    attrs = set(group1.keys())
    if group2:
        attrs = attrs.union(set(group2.keys()))

    if attrs.union(df.columns) != set(df.columns):
        raise InvalidAttributeError(attrs)

    # Form predicate for first group
    pred1 = False
    for attr, vals in group1.items():
        for val in vals:
            pred1 |= df[attr] == val

    # If group2 not given return the predicate and its inverse
    if group2 is None:
        return pred1, ~pred1

    # Form predicate for second group
    pred2 = False
    for attr, vals in group2.items():
        for val in vals:
            pred2 |= df[attr] == val

    return pred1, pred2


def get_predicates_mult(df: pd.DataFrame, groups: List[Dict[str, List[str]]]) -> List[pd.DataFrame]:
    """Similar to get_predicates but works on multiple groups.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        groups (List[Dict[str, List[str]]]):
            A list of groups of interest.

    Raises:
        InvalidAttributeError:
            Indicates an ill-formed group input due to invalid attributes in this case.

    Returns:
        List[pd.DataFrame]:
            A list of dataframes that can be used to index the corresponding groups of data.
    """

    # Check all attributes are valid
    all_attrs = [group.keys() for group in groups]
    attrs = set(*all_attrs)
    if attrs.intersection(df.columns) != attrs:
        raise InvalidAttributeError(attrs)

    # Form a predicate for each group
    preds = []
    for group in groups:
        pred = False
        for attr, vals in group.items():
            for val in vals:
                pred |= df[attr] == val

        preds.append(pred)

    return preds


def fd_opt_bins(column: pd.Series) -> int:
    """Computes the optimal number of bins in a pandas series using the Freedman-Diaconis rule.

    Args:
        column (pd.Series):
            The pandas series containing the continuous data to be binned.

    Returns:
        int:
            The optimal number of bins.
    """

    n = len(column)
    iqr = column.quantile(0.75) - column.quantile(0.25)

    return int((column.max() - column.min()) / (2 * iqr * (n ** (-1 / 3))))
