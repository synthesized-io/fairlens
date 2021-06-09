import json
import os
import pathlib
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as ss

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJ_DIR, "./configs/config_engb.json")

attr_synonym_dict: Dict[str, List[str]] = {}
attr_value_dict: Dict[str, List[str]] = {}


def detect_names_df(
    df: Union[pd.DataFrame, List[str]],
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    deep_search: bool = False,
    config_path: Union[str, pathlib.Path] = None,
) -> Dict[str, Optional[str]]:
    """Detects the sensitive columns in a dataframe or string list and creates a
    dictionary which maps the attribute names to the corresponding sensitive
    category name (such as Gender, Religion etc). The option to deep search can
    be enabled in the case of dataframes, which looks at the values in the tables
    and infers sensitive categories, even when the column name is inconclusive.
    Args:
        df (Union[pd.DataFrame, List[str]]):
            Pandas dataframe or string list that will be analysed.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[Optional[str], Optional[str]], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.
        deep_search (bool, optional):
            The boolean flag that enables deep search when set to true. Deep search
            also makes use of the content of the column to check if it is sensitive.
        config_path (Union[str, pathlib.Path], optional)
            The path of the JSON configuration file in which the dictionaries used for
            detecting sensitive attributes are defined. By default, the configuration
            is the one describing protected attributes and groups according to the
            UK Government.
    Returns:
        Dict[str, Optional[str]]:
            A dictionary containing a mapping from attribute names to a string representing the corresponding
            sensitive attribute category or None.
    Examples:
        >>> detect_names_dict_dataframe(["age", "gender", "legality", "risk"])
        {"age": "Age", "gender": "Gender"}
        >>> col_names = ["native", "location", "house", "subscription", "salary", "religion", "score"]
        >>> df = pd.DataFrame(columns=col_names)
        >>> detect_names_dict_dataframe(df)
        {"native": "Nationality", "location": "Nationality", "house": "Family Status", "religion": "Religion"}
    """
    if config_path:
        attr_synonym_dict, attr_value_dict = load_config(config_path)
    else:
        attr_synonym_dict, attr_value_dict = load_config()

    if isinstance(df, list):
        cols = df
    else:
        cols = df.columns

    sensitive_dict = _detect_names_dict(cols, threshold, str_distance, attr_synonym_dict)

    if isinstance(df, list):
        return sensitive_dict

    str_distance = str_distance or _ro_distance
    sensitive_cols = list(sensitive_dict.keys())

    if deep_search:
        non_sensitive_cols = list(set(cols) - set(sensitive_cols))

        for non_sensitive_col in non_sensitive_cols:
            group_name = _deep_search(df[non_sensitive_col], threshold, str_distance, attr_value_dict)

            if group_name is not None:
                sensitive_dict[non_sensitive_col] = group_name
        return sensitive_dict
    else:
        return sensitive_dict


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
    str_distance = str_distance or _ro_distance

    sensitive_dict = detect_names_df(
        df, threshold=threshold, str_distance=str_distance, deep_search=True, config_path=config_path
    )

    non_sensitive_cols = list(set(df.columns) - set(sensitive_dict.keys()))

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
    str_distance = str_distance or _ro_distance

    sensitive_dict = detect_names_df(
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


def load_config(config_path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> Tuple[Any, Any]:
    """Changes the configuration that creates the underlying synonym and possible value dictionaries
    on which the shallow and deep search algorithms for sensitive attributes are based.

    Args:
        config_path (Union[str, pathlib.Path], optional):
            The path of the JSON file containing the configuration. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        Tuple[Any, Any]:
            Returns a tuple containing the synonym and value dictionaries in a format readable by the
            main detection function.
    """
    with open(config_path) as json_file:
        config_dict = json.load(json_file)

    return config_dict["synonyms"], config_dict["values"]


def _compute_series_correlation(
    sr_a: pd.Series, sr_b: pd.Series, corr_cutoff: float = 0.75, p_cutoff: float = 0.1
) -> bool:
    is_categorical = False
    arrays = list()

    if sr_a.map(type).eq(str).all():
        is_categorical = True

    if sr_b.map(type).eq(str).all():
        if is_categorical:
            # If both columns are categorical, we use Cramer's V.
            if _cramers_v(sr_a, sr_b) > corr_cutoff:
                return True
        else:
            # If just one column is categorical, we can group by it and use Kruskal-Wallis H Test.
            sr_b = sr_b.astype("category").cat.codes
            groups = sr_a.groupby(sr_b)
            for category in sr_b.unique():
                arrays.append(groups.get_group(category))
    else:
        # If just one column is categorical, we can group by it and use Kruskal-Wallis H Test.
        if is_categorical:
            sr_a = sr_a.astype("category").cat.codes
            groups = sr_b.groupby(sr_a)
            for category in sr_a.unique():
                arrays.append(groups.get_group(category))

    # If Kruskal-Wallis has been used, we need to check the p-value instead.
    if arrays:
        args = [group.array for group in arrays]
        try:
            _, p_val = ss.kruskal(*args, nan_policy="omit")
        except ValueError:
            return False
        if p_val < p_cutoff:
            return True

    # If both columns are numeric, we use standard Pearson correlation.
    if abs(sr_a.corr(sr_b)) > corr_cutoff:
        return True

    return False


def _ro_distance(s1: Optional[str], s2: Optional[str]) -> float:
    """Computes a distance between the input strings using the Ratcliff-Obershelp algorithm."""
    if s1 is None or s2 is None:
        return 1

    return 1 - SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _cramers_v(sr_a: pd.Series, sr_b: pd.Series) -> float:
    confusion_matrix = pd.crosstab(sr_a, sr_b)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _detect_name(
    name: str,
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    attr_synonym_dict: Dict[str, List[str]] = None,
) -> Optional[str]:
    """Detects whether a given attribute is sensitive and returns the corresponding sensitive group.
    Args:
        name (str):
            The name of the attribute.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.
        attr_synonym_dict (Dict[str, List[str]]):
            The dictionary of sensitive category synonyms that is used for the shallow search.
            If none is passed, it defaults to the configuration describing protected attributes
            and groups according to the UK Government.
    Returns:
        Optional[str]:
            The sensitive name corresponding to the input.
    """
    if attr_synonym_dict is None:
        attr_synonym_dict, _ = load_config()

    str_distance = str_distance or _ro_distance

    name = name.lower()

    # Check exact match
    for group_name, attrs in attr_synonym_dict.items():
        for attr in attrs:
            if name == attr:
                return group_name

    # Check startswith / endswith
    for group_name, attrs in attr_synonym_dict.items():
        for attr in attrs:
            if name.startswith(attr) or name.endswith(attr):
                return group_name

    # Check distance < threshold
    for group_name, attrs in attr_synonym_dict.items():
        for attr in attrs:
            dist = str_distance(name, attr)
            if dist < threshold:
                return group_name

    return None


def _detect_names_dict(
    names: List[str],
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    attr_synonym_dict: Dict[str, List[str]] = None,
) -> Dict[str, Optional[str]]:
    """Creates a dictionary which maps the attribute names to the corresponding sensitive attribute.
    Args:
        names (List[str]):
            List of attribute names.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.
        attr_synonym_dict (Dict[str, List[str]]):
            The dictionary of sensitive category synonyms that is used for the shallow search.
            If none is passed, it defaults to the configuration describing protected attributes
            and groups according to the UK Government.
    Returns:
        Dict[str, Optional[str]]:
            A dictionary containing a mapping from attribute names to a string representing the corresponding
            sensitive attribute category or None.
    Examples:
        >>> _detect_names_dict(["age", "gender", "legality", "risk"])
        {"age": "Age", "gender": "Gender", "legality": None, "risk": None}
    """
    if attr_synonym_dict is None:
        attr_synonym_dict, _ = load_config()

    names_dict = dict()

    for name in names:
        names_dict[name] = _detect_name(
            name, threshold=threshold, str_distance=str_distance, attr_synonym_dict=attr_synonym_dict
        )

    # Remove columns with 'None' values.
    for key, value in dict(names_dict).items():
        if value is None:
            del names_dict[key]

    return names_dict


def _deep_search(
    s: pd.Series,
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    attr_value_dict: Dict[str, List[str]] = None,
) -> Optional[str]:
    if attr_value_dict is None:
        _, attr_value_dict = load_config()

    # Avoid checking number values as they can be inconclusive.
    if s.dtype.kind in ["i", "f", "m", "M"]:
        return None

    str_distance = str_distance or _ro_distance

    # Coarse grain search to check if there is an exact match to avoid mismatches.
    for group_name, values in attr_value_dict.items():
        # Skip sensitive groups that do not have defined possible values.
        if not values:
            continue
        pattern = "|".join(values)
        if s.isin(values).mean() > 0.2:
            return group_name

    for group_name, values in attr_value_dict.items():
        if not values:
            continue
        pattern = "|".join(values)
        if s.str.contains(pattern).mean() > 0.6:
            return group_name

    # Fine grain search that will catch edge cases.
    for group_name, values in attr_value_dict.items():
        for value in values:
            if s.map(lambda x: str_distance(x, value) < threshold).mean() > 0.1:
                return group_name
    return None
