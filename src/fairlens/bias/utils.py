from typing import Optional

import numpy as np
import pandas as pd


def bin(
    df: pd.DataFrame,
    column_name: str,
    n_bins: int,
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
        n_bins (int):
            The number of bins.
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

    df.loc[:, column] = pd.cut(df.loc[:, column_name], bins=bins, **kwargs)

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

    in_dtype = str(df[column_name].dtype)

    column = df[column_name].copy()

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
