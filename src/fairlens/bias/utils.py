import calendar
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class DistrType(Enum):
    """Indicates the type distribution of data in a series."""

    Continuous = "continuous"
    Binary = "binary"
    Categorical = "categorical"
    Datetime = "datetime"

    def is_continuous(self):
        return self == DistrType.Continuous

    def is_binary(self):
        return self == DistrType.Binary

    def is_categorical(self):
        return self == DistrType.Categorical

    def is_datetime(self):
        return self == DistrType.Datetime


def zipped_hist(
    data: Tuple[pd.Series, ...],
    bin_edges: Optional[np.ndarray] = None,
    normalize: bool = True,
    ret_bins: bool = False,
    distr_type: Optional[str] = None,
) -> Union[Tuple[pd.Series, ...], Tuple[Tuple[pd.Series, ...], Optional[np.ndarray]]]:
    """Bins a tuple of series' and returns the aligned histograms.

    Args:
        data (Tuple[pd.Series, ...]):
            A tuple consisting of the series' to be binned. All series' must have the same dtype.
        bin_edges (Optional[np.ndarray], optional):
            Bin edges to bin continuous data by. Defaults to None.
        normalize (bool, optional):
            Normalize the histograms, turning them into pdfs. Defaults to True.
        ret_bins (bool, optional):
            Returns the bin edges used in the histogram. Defaults to False.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.

    Returns:
        Union[Tuple[np.ndarray, ...], Tuple[Tuple[np.ndarray, ...], Optional[np.ndarray]]]:
            A tuple of np.ndarrays consisting of each histogram for the input data.
            Additionally returns bins if ret_bins is True.
    """

    joint = pd.concat(data)
    is_continuous = distr_type == "continuous" if distr_type is not None else infer_distr_type(joint).is_continuous()

    # Compute histograms of the data, bin if continuous
    if is_continuous:
        # Compute shared bin_edges if not given, and use np.histogram to form histograms
        if bin_edges is None:
            bin_edges = np.histogram_bin_edges(joint, bins="auto")

        hists = [np.histogram(series, bins=bin_edges)[0] for series in data]

        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                hists = [np.nan_to_num(hist / hist.sum()) for hist in hists]

    else:
        # For categorical data, form histogram using value counts and align
        space = joint.unique()

        dicts = [sr.value_counts(normalize=normalize) for sr in data]
        hists = [np.array([d.get(val, 0) for val in space]) for d in dicts]

    ps = [pd.Series(hist) for hist in hists]

    if ret_bins:
        return tuple(ps), bin_edges

    return tuple(ps)


def bin(
    column: pd.Series,
    n_bins: Optional[int] = None,
    remove_outliers: Optional[float] = 0.1,
    quantile_based: bool = False,
    bin_centers=False,
    **kwargs,
) -> pd.Series:
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
        bin_centers (bool, optional):
            Return the mean of the intervals instead of the intervals themselves. Defaults to False.
        **kwargs:
            Key word arguments for pd.cut or pd.qcut.

    Returns:
        pd.Series:
            The binned column.
    """

    column = infer_dtype(column)
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

    if bin_centers:
        binned = binned.apply(lambda i: i.mid)

    return binned


def quantize_date(column: pd.Series):
    """Quantize a column of dates into bins of uniform width in years, days or months.

    Args:
        column (pd.Series):
            The column of dates to quantize. Must be have a dtype of "datetime64[ns]".
    """

    TEN_YEAR_THRESHOLD = 15
    TEN_MIN_THRESHOLD = 15
    TEN_SEC_THRESHOLD = 15

    if column.dtype != "datetime64[ns]":
        raise ValueError("'quantize_date' requires the column to be a pandas datetime object")

    years, months, days = column.dt.year, column.dt.month, column.dt.day
    hours, minutes, seconds = column.dt.hour, column.dt.minute, column.dt.second

    # Assuming dates don't go back beyond a 100 years.
    if years.max() - years.min() >= TEN_YEAR_THRESHOLD:
        return ((years // 10) * 10).apply(lambda x: str(x) + "-" + str(x + 10))
    elif years.nunique() > 1:
        return years
    elif months.nunique() > 1:
        return months.apply(lambda x: calendar.month_abbr[x])
    elif days.nunique() > 1:
        return days.apply(lambda x: "Day " + str(x))
    elif hours.nunique() > 1:
        return hours.apply(lambda x: "Hour " + str(x))
    elif minutes.max() - minutes.min() > TEN_MIN_THRESHOLD:
        return ((minutes // 10) * 10).apply(lambda x: str(x) + "min-" + str(x + 10) + "min")
    elif minutes.nunique() > 1:
        return minutes.apply(lambda x: str(x) + "m")
    elif seconds.max() - seconds.min() > TEN_SEC_THRESHOLD:
        return ((seconds // 10) * 10).apply(lambda x: str(x) + "s-" + str(x + 10) + "s")

    return seconds


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
    if column.dtype.kind not in ("i", "u", "f", "M"):
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

    col = infer_dtype(column)

    unique = col.unique()
    n_unique = len(unique)
    n_rows = len(col)
    dtype = col.dtype

    if dtype == "datetime64[ns]":
        return DistrType.Datetime

    elif n_unique > max(min_num_unique, ctl_mult * np.log(n_rows)) and dtype in ["float64", "int64"]:
        return DistrType.Continuous

    elif n_unique == 2 and np.isin(unique, [0, 1]).all():
        return DistrType.Binary

    else:
        return DistrType.Categorical


def get_predicates_mult(
    df: pd.DataFrame, groups: Sequence[Union[Mapping[str, List[Any]], pd.Series]]
) -> List[pd.Series]:
    """Similar to get_predicates but works on multiple groups.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        groups (Sequence[Union[Mapping[str, List[Any]], pd.Series]]):
            A list of groups of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
            Examples: {"Sex": ["Male"]}, df["Sex"] == "Female"

    Raises:
        ValueError:
            Indicates an ill-formed group or invalid input group.

    Returns:
        List[pd.Series]:
            A list of series' that can be used to index the corresponding groups of data.
    """

    predicates = {}
    remaining_groups = []

    # Separate groups for which predicates are to be computed.
    for i, group in enumerate(groups):
        if isinstance(group, dict):
            remaining_groups.append((i, group))
        else:
            if not isinstance(group, pd.Series) or group.dtype != "bool":
                raise ValueError(
                    "Invalid group detected. Groups must be either dictionaries or pandas series' of bools."
                )

            predicates[i] = group

    # Check all attributes are valid
    all_attrs = [group.keys() for _, group in remaining_groups]
    attrs = set().union(*all_attrs)  # type: ignore

    if attrs.intersection(df.columns) != attrs:
        raise ValueError(f"Invalid attribute detected. Attributes must be in:\n{df.columns}")

    # Form a predicate for each remaining group
    for i, group in remaining_groups:
        predicates[i] = df[group.keys()].isin(group).all(axis=1)

    return [pred for _, pred in sorted(predicates.items(), key=lambda p: p[0])]


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
