"""
This module contains statistical measures for analyzing target variable distributions across sensitive groups.
"""

import functools
import operator
from typing import Any, List, Mapping, Sequence, Union

import pandas as pd
from scipy.stats import describe, entropy

from .. import utils


def _mean_numerical(x: pd.Series) -> float:
    return describe(x).mean


def _variance_numerical(x: pd.Series) -> float:
    return describe(x).variance


def _mean_datetime(x: pd.Series) -> pd.Timedelta:
    nums = pd.to_datetime(x)
    date_min = nums.min()
    diffs = [num - date_min for num in nums]
    date_mean = date_min + functools.reduce(operator.add, diffs) / len(diffs)
    return date_mean


def _variance_datetime(x: pd.Series) -> pd.Timedelta:
    nums = pd.to_datetime(x).astype(int)
    res = nums.std()
    std = pd.to_timedelta(res)
    return std


def _mode_categorical(x: pd.Series) -> Any:
    return x.value_counts(sort=True).index[0]


def _variance_square_sum(x: pd.Series) -> float:
    return (x.value_counts(normalize=True) ** 2).sum()


def _variance_entropy(x: pd.Series) -> float:
    counts = x.value_counts()
    return entropy(counts)


def _means_multinomial(x: pd.Series) -> pd.Series:
    return x.value_counts(normalize=True, sort=False)


def _variances_multinomial(x: pd.Series) -> pd.Series:
    probs = x.value_counts(normalize=True, sort=False)
    variances = pd.Series([prob * (1 - prob) for prob in probs], index=probs.index)
    return variances


def sensitive_group_analysis(
    df: pd.DataFrame,
    target_attr: str,
    groups: Sequence[Union[Mapping[str, List[Any]], pd.Series]],
    categorical_mode: str = "multinomial",
) -> pd.DataFrame:
    """This function produces a summary of the first two central moments of the distributions created
    from the target attribute by applying predicates generated by a list of groups of interest. Allows
    the user to quickly scan how the target varies and how the expected value is different based on
    possibly protected attributes.
    Supports binary, date-like, numerical and categorical data for the target column.

    Args:
        df (pd.DataFrame):
            The input datafame.
        target_attr (str):
            The target attribute in the dataframe from which the distributions are formed.
        groups (List[Union[Mapping[str, List[Any]], pd.Series]]):
            The list of groups of interest. Each group can be a mapping / dict from attribute to value or
            a predicate itself, i.e. pandas series consisting of bools which can be used as a predicate
            to index a subgroup from the dataframe.
            Examples of valid groups: {"Sex": ["Male"]}, df["Sex"] == "Female"
        categorical_mode (str):
            Allows the user to choose which method will be used for computing the first moment for categorical
            (and implicitly, binary) series. Can be "square", "entropy" which will use the mode or "multinomial",
            which returns the probability of each variable occuring. Defaults to "multinomial".

    Returns:
        pd.DataFrame:
            A dataframe comprising and reporting the results for the means and variances across the groups
            of interest which is adapted to the type of the underlying data in the target column.
    """

    preds = utils.get_predicates_mult(df, groups)
    distrs = [df[pred][target_attr] for pred in preds]
    target_type = utils.infer_distr_type(df[target_attr])

    if target_type.is_continuous():
        sr_type = "continuous"
    elif target_type.is_datetime():
        sr_type = "datetime"
    else:
        sr_type = "categorical"

    means = [compute_distribution_mean(distr, x_type=sr_type, categorical_mode=categorical_mode) for distr in distrs]
    variances = [
        compute_distribution_variance(distr, x_type=sr_type, categorical_mode=categorical_mode) for distr in distrs
    ]

    # In the case of the multinomial mode of analysis for the categorical variable, the output results from
    # the corresponding functions for the mean and variance will output series instead of floats (as they
    # compute a mean and variance for each of the nominal variables).
    # We create two dataframes, one for means and one for variances, where the column names refer to
    # the categorical variables and the indexes refer to the corresponding groups.
    if target_type.is_categorical() and categorical_mode == "multinomial":
        means_df = pd.DataFrame(means, means.index, columns=df[target_attr].value_counts(sort=False))
        variances_df = pd.DataFrame(variances, variances.index, columns=df[target_attr].value_counts(sort=False))

        return means_df.append(variances_df)

    results = {"Means": means, "Variances": variances}

    return pd.DataFrame(results, index=groups)


def compute_distribution_mean(
    x: pd.Series, x_type: str, categorical_mode: str = "multinomial"
) -> Union[float, pd.Series]:
    """This function computes the mean (means) of a given distribution, based on the type of its underlying
    data. Supports binary, date-like, numerical and categorical data for the distribution.

    Args:
        x (pd.Series):
            The series representing the distribution for which the mean will be calculated
        x_type (str):
            This is the underlying type of the target attribute distribution and is passed to avoid errors caused
            by very specific groping.
        categorical_mode (str, optional):
            Allows the user to choose which method will be used for computing the first moment for categorical
            (and implicitly, binary) series. Can be "square", "entropy" which will use the mode or "multinomial",
            which returns the probability of each variable occuring. Defaults to "multinomial".

    Returns:
        Union[float, pd.Series]:
            The mean (or means, if considering a categorical distribution to be multinomial, for example)
            of the given distribution.
    """

    if x_type == "continuous":
        return _mean_numerical(x)
    elif x_type == "datetime":
        return _mean_datetime(x)
    elif categorical_mode == "square":
        return _mode_categorical(x)
    elif categorical_mode == "entropy":
        return _mode_categorical(x)
    elif categorical_mode == "multinomial":
        return _means_multinomial(x)
    else:
        return None


def compute_distribution_variance(
    x: pd.Series, x_type: str, categorical_mode: str = "multinomial"
) -> Union[float, pd.Series]:
    """This function computes the variances (variances) of a given distribution, based on the type of its underlying
    data. Supports binary, date-like, numerical and categorical data for the distribution.

    Args:
        x (pd.Series):
            The series representing the distribution for which the variance will be calculated
        x_type (str):
            This is the underlying type of the target attribute distribution and is passed to avoid errors caused
            by very specific groping.
        categorical_mode (str, optional):
            Allows the user to choose which method will be used for computing the first moment for categorical
            (and implicitly, binary) series. Can be "square", "entropy" which will use the mode or "multinomial",
            which returns the probability of each variable occuring. Defaults to "multinomial".

    Returns:
        Union[float, pd.Series]:
            The variance (or variances if considering a categorical distribution to be multinomial, for example)
            of the given distribution.
    """

    if x_type == "continuous":
        return _variance_numerical(x)
    elif x_type == "datetime":
        return _variance_datetime(x)
    elif categorical_mode == "square":
        return _variance_square_sum(x)
    elif categorical_mode == "entropy":
        return _variance_entropy(x)
    elif categorical_mode == "multinomial":
        return _variances_multinomial(x)
    else:
        return None
