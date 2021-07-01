from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


def histogram(
    data: Tuple[pd.Series, ...],
    bin_edges: Optional[np.ndarray] = None,
    normalize: bool = True,
    ret_bins: bool = False,
    distr_type: Optional[str] = None,
) -> Union[Tuple[pd.Series, ...], Tuple[Tuple[pd.Series, ...], Optional[np.ndarray]]]:
    """Bins a tuple of series'= and returns the aligned histograms.

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

        dicts = [series.value_counts(normalize=normalize) for series in data]
        hists = [np.array([d.get(val, 0) for val in space]) for d in dicts]

    ps = [pd.Series(hist) for hist in hists]

    if ret_bins:
        return tuple(ps), bin_edges

    return tuple(ps)


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

    col = infer_dtype(column)

    unique = col.unique()
    n_unique = len(unique)
    n_rows = len(col)
    dtype = col.dtype

    if n_unique > max(min_num_unique, ctl_mult * np.log(n_rows)) and dtype in ["float64", "int64"]:
        return DistrType.Continuous

    elif n_unique == 2 and np.isin(unique, [0, 1]).all():
        return DistrType.Binary

    else:
        return DistrType.Categorical


def get_predicates_mult(df: pd.DataFrame, groups: Sequence[Dict[str, List[str]]]) -> List[pd.Series]:
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
