import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as ss

from fairlens.sensitive import detection as dt


def find_sensitive_correlations(
    df: pd.DataFrame,
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    corr_cutoff: float = 0.75,
    p_cutoff: float = 0.1,
    config_path: Union[str, pathlib.Path] = None,
) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """Looks at the columns that are not considered to be immediately sensitive and finds if any is strongly
    correlated with a sensitive column, specifying both the sensitive column name and the sensitive category
    it is a part of.

    Args:
        df (pd.DataFrame):
            Pandas dataframe that will be analyzed.
        threshold (float, optional):
            The threshold for the string distance function that will be used for detecting sensitive columns.
            Defaults to 0.1.
        str_distance (Callable[[Optional[str], Optional[str]], float], optional):
            The string distance function that will be used for detecting sensitive columns.
            Defaults to Ratcliff-Obershelp algorithm.
        corr_cutoff (float, optional):
            The cutoff for considering a column to be correlated with a sensitive attribute, with Pearson's correlation.
            Defaults to 0.75.
        p_cutoff (float, optional):
            The p-value cutoff to be used when checking if a categorical column is correlated with a numeric column
            using the Kruskal-Wallis H Test.
        config_path (Union[str, pathlib.Path], optional)
            The path of the JSON configuration file in which the dictionaries used for
            detecting sensitive attributes are defined. By default, the configuration
            is the one describing protected attributes and groups according to the
            UK Government.

    Returns:
        Dict[str, Tuple[Optional[str]]]:
            The returned value is a dictionary with the non-sensitive column as the key and a tuple as the value,
            where the first entry is the name of the corresponding sensitive column in the dataframe and the second
            entry is the sensitive attribute category.
    """
    str_distance = str_distance or dt._ro_distance

    sensitive_dict = dt.detect_names_df(
        df, threshold=threshold, str_distance=str_distance, deep_search=True, config_path=config_path
    )

    non_sensitive_cols = df.columns.difference(sensitive_dict)

    correlation_dict = dict()

    for non_sensitive_col in non_sensitive_cols:
        col1 = df[non_sensitive_col]

        correlation_list = list()

        for sensitive_col in sensitive_dict.keys():
            col2 = df[sensitive_col]

            if _compute_series_correlation(col1, col2, corr_cutoff, p_cutoff):
                correlation_list.append((sensitive_col, sensitive_dict[sensitive_col]))

        if len(correlation_list) > 0:
            correlation_dict[non_sensitive_col] = correlation_list

    return correlation_dict


def find_column_correlation(
    col: Union[str, pd.Series],
    df: pd.DataFrame,
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    corr_cutoff: float = 0.75,
    p_cutoff: float = 0.1,
    config_path: Union[str, pathlib.Path] = None,
) -> List[Tuple[str, Optional[str]]]:
    """This function takes in a series or a column name of a given dataframe and checks whether any of
    the sensitive attribute columns detected in the dataframe are strongly correlated with the series
    or the column corresponding to the given name.
    If matches are found, a list containing the correlated
    column names and its associated sensitive category, respectively, is returned.

    Args:
        col (Union[str, pd.Series]):
            Pandas series or dataframe column name that will be analyzed.
        df (pd.DataFrame):
            Dataframe supporting the search, possibly already a column with the input name.
        threshold (float, optional):
            The threshold for the string distance function that will be used for detecting sensitive columns.
            Defaults to 0.1.
        str_distance (Callable[[Optional[str], Optional[str]], float], optional):
            The string distance function that will be used for detecting sensitive columns.
            Defaults to Ratcliff-Obershelp algorithm.
        corr_cutoff (float, optional):
            The cutoff for considering a column to be correlated with a sensitive attribute, with Pearson's correlation.
            Defaults to 0.75.
        p_cutoff (float, optional):
            The p-value cutoff to be used when checking if a categorical column is correlated with a numeric column
            using the Kruskal-Wallis H Test.
        config_path (Union[str, pathlib.Path], optional)
            The path of the JSON configuration file in which the dictionaries used for
            detecting sensitive attributes are defined. By default, the configuration
            is the one describing protected attributes and groups according to the
            UK Government.

    Returns:
        List[Tuple[str, Optional[str]]]:
            The returned value is a list containing tuples of all the correlated sensitive columns that were
            found, along with their associated sensitive category label.
    """
    str_distance = str_distance or dt._ro_distance

    sensitive_dict = dt.detect_names_df(
        df, threshold=threshold, str_distance=str_distance, deep_search=True, config_path=config_path
    )

    correlation_list = list()

    if isinstance(col, str):
        if col in df.columns:
            col1 = df[col]
        else:
            raise ValueError("The given dataframe does not contain a column with this name.")
    else:
        col1 = col

    for sensitive_col in sensitive_dict.keys():
        col2 = df[sensitive_col]

        if _compute_series_correlation(col1, col2, corr_cutoff, p_cutoff):
            correlation_list.append((sensitive_col, sensitive_dict[sensitive_col]))

    return correlation_list


def _compute_series_correlation(
    sr_a: pd.Series, sr_b: pd.Series, corr_cutoff: float = 0.75, p_cutoff: float = 0.1
) -> bool:
    a_categorical = sr_a.map(type).eq(str).all()
    b_categorical = sr_b.map(type).eq(str).all()
    arrays = list()

    if a_categorical and b_categorical:
        # If both columns are categorical, we use Cramer's V.
        if _cramers_v(sr_a, sr_b) > corr_cutoff:
            return True
    elif not a_categorical and b_categorical:
        # If just one column is categorical, we can group by it and use Kruskal-Wallis H Test.
        sr_b = sr_b.astype("category").cat.codes
        groups = sr_a.groupby(sr_b)
        arrays = [groups.get_group(category) for category in sr_b.unique()]
    elif a_categorical and not b_categorical:
        # If just one column is categorical, we can group by it and use Kruskal-Wallis H Test.
        sr_a = sr_a.astype("category").cat.codes
        groups = sr_b.groupby(sr_a)
        arrays = [groups.get_group(category) for category in sr_a.unique()]

    # If we have a categorical-continuous association, we use Kruskal-Wallis and check the p-value instead.
    if arrays:
        args = [np.array(group.array, dtype=float) for group in arrays]
        try:
            _, p_val = ss.kruskal(*args, nan_policy="omit")
        except ValueError:
            return False
        if p_val < p_cutoff:
            return True

    # If both columns are numeric, we use standard Pearson correlation and the correlation cutoff.
    return abs(sr_a.corr(sr_b)) > corr_cutoff


def _cramers_v(sr_a: pd.Series, sr_b: pd.Series) -> float:
    confusion_matrix = pd.crosstab(sr_a, sr_b)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = len(sr_a)
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
