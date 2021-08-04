"""
Identify legally protected attributes in a dataset.
"""

import json
import os
import pathlib
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJ_DIR, "./configs/config_engb.json")

attr_synonym_dict: Dict[str, List[str]] = {}
attr_value_dict: Dict[str, List[str]] = {}


def detect_names_df(
    df: Union[pd.DataFrame, List[str]],
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    deep_search: bool = False,
    n_samples: int = 20,
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
        n_samples (int, optional):
            The number of values to be sampled from series of large datasets when using
            the deep search algorithm. A low sample number will greatly improve speed and
            still produce accurate results, assuming that the underlying dictionaries are
            comprehensive.
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

    str_distance = str_distance or _ro_distance

    if isinstance(df, list):
        cols = df
        return _detect_names_dict(
            cols, threshold=threshold, str_distance=str_distance, attr_synonym_dict=attr_synonym_dict
        )
    else:
        cols = df.columns

    sensitive_dict = _detect_names_dict(cols, threshold, str_distance, attr_synonym_dict)

    if deep_search:

        for col in cols:
            # Series containing only the unique values of the analyzed column.
            uniques = pd.Series(df[col].unique())
            # If the series are larger than the provided n_samples, we take a sample to increase speed.
            if uniques.size > n_samples:
                column = uniques.sample(n=n_samples)
            else:
                column = uniques

            group_name = _deep_search(column, threshold, str_distance, attr_value_dict)

            if group_name is not None:
                sensitive_dict[col] = group_name

    return sensitive_dict


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


def _ro_distance(s1: Optional[str], s2: Optional[str]) -> float:
    """Computes a distance between the input strings using the Ratcliff-Obershelp algorithm."""
    if s1 is None or s2 is None:
        return 1

    if not isinstance(s1, str) or not isinstance(s2, str):
        return 1

    return 1 - SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


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
        separator = " ,.-:"
        for attr in attrs:
            if name.startswith(attr.lower() + "|".join(separator)) or name.endswith("|".join(separator) + attr.lower()):
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
