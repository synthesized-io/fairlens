from enum import Enum
from typing import Dict, List, Optional

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
    column: pd.Series,
    n_bins: Optional[int] = None,
    remove_outliers: Optional[float] = 0.1,
    quantile_based: bool = False,
    mean_bins=False,
    **kwargs
) -> pd.DataFrame:
    """Bin continous values into discrete bins.

    Args:
        column (pd.Series):
            The column or series containing the data to be binned.
        n_bins (Optional[int], optional):
            The number of bins. Defaults to Freedman-Diaconis rule.
        remove_outliers (Optional[float], optional):
            Any data point outside this quantile (two-sided) will be dropped before computing bins.
            If `None`, outliers are not removed. Defaults to 0.1.
        quantile_based (bool, optional):
            Whether the bin computation is quantile based. Defaults to False.
        mean_bins (bool, optional):
            Return the mean of the intervals instead of the intervals themselves. Defaults to False.
        **kwargs:
            Key word arguments for pd.cut or pd.qcut.

    Returns:
        pd.Series:
            The binned column.
    """

    column = column.copy()
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

    binned = pd.Series(pd.cut(column, bins=bins, include_lowest=True, **kwargs))

    if mean_bins:
        return binned.apply(lambda i: float(i.mid))

    return binned


def infer_dtype(col: pd.Series) -> pd.Series:
    """Infers the type of the data and converts the data to it.

    Args:
        col (str):
            The column of the dataframe to transform.

    Returns:
        pd.Series:
            The column converted to its inferred type.
    """

    column = col.copy()

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
        return col
    elif out_dtype in ("i", "u", "f", "f8", "i8", "u8"):
        return pd.to_numeric(col, errors="coerce")

    return col.astype(out_dtype, errors="ignore")


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

    unique = column.unique()
    n_unique = len(unique)
    n_rows = len(column)
    dtype = infer_dtype(column).dtype

    if n_unique > max(min_num_unique, ctl_mult * np.log(n_rows)) and dtype in ["float64", "int64"]:
        return DistrType.Continuous

    elif (n_unique == 2 and 0 in unique and 1 in unique) or (unique == [1]).all() or (unique == [0]).all():
        return DistrType.Binary

    else:
        return DistrType.Categorical


def get_predicates_mult(df: pd.DataFrame, groups: List[Dict[str, List[str]]]) -> List[pd.Series]:
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
        List[pd.Series]:
            A list of series' that can be used to index the corresponding groups of data.
    """

    # Check all attributes are valid
    all_attrs = [group.keys() for group in groups]
    attrs = set().union(*all_attrs)  # type: ignore

    if attrs.intersection(df.columns) != attrs:
        raise InvalidAttributeError(attrs)

    # Form a predicate for each group
    preds = []
    for group in groups:
        pred = df[group.keys()].isin(group).all(axis=1)
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


def get_all_subclasses(cls):
    if len(cls.__subclasses__()) == 0:
        return [cls]

    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses
