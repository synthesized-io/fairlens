from difflib import SequenceMatcher
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


class SensitiveNames(Enum):
    """
    An enum of sensitive attributes to be detected.
    """

    Age: str = "Age"
    Gender: str = "Gender"
    Ethnicity: str = "Ethnicity"
    Religion: str = "Religion"
    Nationality: str = "Nationality"
    FamilyStatus: str = "Family Status"
    Disability: str = "Disability"
    SexualOrientation: str = "Sexual Orientation"


sensitive_names_map: Dict["SensitiveNames", List[str]] = {
    SensitiveNames.Age: ["age", "DOB", "birth", "youth", "elder", "senior"],
    SensitiveNames.Gender: ["gender", "sex"],
    SensitiveNames.Ethnicity: ["race", "color", "ethnic", "breed", "culture"],
    SensitiveNames.Nationality: ["nation", "geography", "location", "native", "country", "region"],
    SensitiveNames.Religion: ["religion", "creed", "cult", "doctrine"],
    SensitiveNames.FamilyStatus: ["family", "house", "marital", "children", "partner", "pregnant"],
    SensitiveNames.Disability: ["disability", "impairment"],
    SensitiveNames.SexualOrientation: ["sexual", "orientation", "attracted"],
}

sensitive_values_map: Dict["SensitiveNames", List[str]] = {
    SensitiveNames.Age: [],
    SensitiveNames.Gender: [
        "male",
        "female",
        "gender",
        "non-binary",
        "man",
        "woman",
        "gender neutral",
        "agender",
        "pangender",
        "genderqueer",
        "two-spirit",
    ],
    SensitiveNames.Ethnicity: [
        "caucasian",
        "american",
        "african",
        "indo-european",
        "asian",
        "caribbean",
        "arabian",
        "indigenous",
        "romani",
        "white",
        "black",
        "indian",
        "romance",
    ],
    SensitiveNames.Nationality: [
        "english",
        "irish",
        "welsh",
        "american",
        "german",
        "indian",
        "french",
        "spanish",
        "indian",
        "chinese",
        "japanese",
        "afghan",
        "australian",
        "albanian",
        "argentinian",
        "austrian",
        "bangladeshi",
        "bengali",
        "belgian",
        "bolivian",
        "botswanan",
        "brazilian",
        "bulgarian",
        "cambodian",
        "cameroonian",
        "canadian",
        "chilean",
        "colombian",
        "costa rican",
        "croatian",
        "cuban",
        "czech",
        "danish",
        "dane",
        "dominican",
        "ecuadorian",
        "egyptian",
        "salvadorian",
        "estonian",
        "ethiopian",
        "fijian",
        "finnish",
        "ghanaian",
        "greek",
        "guatemalan",
        "haitian",
        "honduran",
        "hungarian",
        "icelandic",
        "indonesian",
        "iranian",
        "iraqi",
        "israeli",
        "italian",
        "jamaican",
        "jordanian",
        "kenyan",
        "kuwaiti",
        "laotain",
        "lao",
        "latvian",
        "lebanese",
        "libyan",
        "lithuanian",
        "malagasy",
        "malaysian",
        "malian",
        "maltese",
        "mexican",
        "mongolian",
        "moroccan",
        "mozambican",
        "nambian",
        "nepalese",
        "dutch",
        "new zealander",
        "nicaraguan",
        "nigerian",
        "norwegian",
        "pakistani",
        "panamanian",
        "paraguayan",
        "peruvian",
        "filipino",
        "polish",
        "portuguese",
        "romanian",
        "russian",
        "saudi",
        "scottish",
        "senegalese",
        "serbian",
        "singaporean",
        "slovak",
        "south african",
        "korean",
        "sri lankan",
        "sudanese",
        "swedish",
        "swiss",
        "syrian",
        "taiwanese",
        "taijikistani",
        "thai",
        "tongan",
        "tunisian",
        "turkish",
        "ukrainian",
        "emirati",
        "uruguayan",
        "venezuelan",
        "vietnamese",
        "zambian",
        "zimbabwean",
    ],
    SensitiveNames.Religion: [
        "christian",
        "religion",
        "islam",
        "hinduism",
        "sikhism",
        "judaism",
        "buddhism",
        "protestantism",
        "daoism",
        "pantheism",
    ],
    SensitiveNames.FamilyStatus: ["married", "single", "widowed", "divorced", "separated", "pregnant"],
    SensitiveNames.Disability: [
        "hearing",
        "impairment",
        "sight",
        "cancer",
        "hiv",
        "sclerosis",
        "injury",
        "learning",
        "dyslexia",
        "dyspraxia",
        "autism",
        "depression",
        "disorder",
        "bipolar",
        "schizophrenia",
        "arthritis",
        "fibromyalgia",
        "dystrophy",
        "dementia",
    ],
    SensitiveNames.SexualOrientation: ["heterosexual", "asexual", "homosexual", "bisexual", "sexual"],
}


def detect_names_df(
    df: Union[pd.DataFrame, List[str]],
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    deep_search: bool = False,
) -> Dict[str, Optional[str]]:
    """Detects the sensitive columns in a dataframe or string list and creates a
    dictionary which maps the attribute names to the corresponding sensitive
    category name (such as Gender, Religion etc). The option to deep search can
    be enabled in the case of dataframes, which looks at the values in the tables
    and infers sensitive categories, even when the column name is inconclusive.

    Args:
        df Union[pd.DataFrame, List[str]]):
            Pandas dataframe or string list that will be analysed.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[Optional[str], Optional[str]], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.
        deep_search (bool, optional):
            The boolean flag that enables deep search when set to true. Deep search
            also makes use of the content of the column to check if it is sensitive.

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
    if isinstance(df, list):
        cols = df
    else:
        cols = df.columns

    sensitive_dict = _detect_names_dict(cols, threshold, str_distance)

    if isinstance(df, list):
        return sensitive_dict

    str_distance = str_distance or _ro_distance
    sensitive_cols = list(sensitive_dict.keys())

    if deep_search:
        non_sensitive_cols = list(set(cols) - set(sensitive_cols))

        for non_sensitive_col in non_sensitive_cols:
            group_name = _deep_search(df[non_sensitive_col], threshold, str_distance)

            if group_name is not None:
                sensitive_dict[non_sensitive_col] = group_name
        return sensitive_dict
    else:
        return sensitive_dict


def detect_correlation(
    df: pd.DataFrame,
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    corr_cutoff: float = 0.75,
) -> Dict[str, Tuple[str, Optional[str]]]:
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

    Returns:
        Dict[str, Tuple[Optional[str]]]:
            The returned value is a dictionary with the non-sensitive column as the key and a tuple as the value,
            where the first entry is the name of the corresponding sensitive column in the dataframe and the second
            entry is the sensitive attribute category.
    """
    str_distance = str_distance or _ro_distance

    sensitive_dict = detect_names_df(df, threshold=threshold, str_distance=str_distance, deep_search=True)

    non_sensitive_cols = list(set(df.columns) - set(sensitive_dict.keys()))

    correlation_dict = dict()

    for sensitive_col in sensitive_dict.keys():
        for non_sensitive_col in non_sensitive_cols:
            col1 = df[sensitive_col]
            col2 = df[non_sensitive_col]

            # Turn string columns into numerical representation to be able to correlate.
            if df[sensitive_col].map(type).eq(str).all():
                col1 = df[sensitive_col].astype("category").cat.codes
            if df[non_sensitive_col].map(type).eq(str).all():
                col2 = df[sensitive_col].astype("category").cat.codes

            if abs(col1.corr(col2)) > corr_cutoff:
                correlation_dict[non_sensitive_col] = (sensitive_col, sensitive_dict[sensitive_col])

    return correlation_dict


def check_column_correlation(
    col: Union[str, pd.Series],
    df: pd.DataFrame,
    threshold: float = 0.1,
    str_distance: Callable[[Optional[str], Optional[str]], float] = None,
    corr_cutoff: float = 0.75,
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

    Returns:
        List[Tuple[str, Optional[str]]]:
            The returned value is a list containing tuples of all the correlated sensitive columns that were
            found, along with their associated sensitive category label.
    """
    str_distance = str_distance or _ro_distance

    sensitive_dict = detect_names_df(df, threshold=threshold, str_distance=str_distance, deep_search=True)

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

        # Turn string columns into numerical representations.
        if col1.map(type).eq(str).all():
            col1 = col1.astype("category").cat.codes
        if df[sensitive_col].map(type).eq(str).all():
            col2 = col2.astype("category").cat.codes

        if abs(col1.corr(col2)) > corr_cutoff:
            correlation_list.append((sensitive_col, sensitive_dict[sensitive_col]))

    return correlation_list


def _ro_distance(s1: Optional[str], s2: Optional[str]) -> float:
    """Computes a distance between the input strings using the Ratcliff-Obershelp algorithm."""
    if s1 is None or s2 is None:
        return 1

    return 1 - SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _detect_name(
    name: str, threshold: float = 0.1, str_distance: Callable[[Optional[str], Optional[str]], float] = None
) -> Optional[str]:
    """Detects whether a given attribute is sensitive and returns the corresponding sensitive group.

    Args:
        name (str):
            The name of the attribute.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.

    Returns:
        Optional[str]:
            The sensitive name corresponding to the input.
    """

    str_distance = str_distance or _ro_distance

    name = name.lower()

    # Check exact match
    for group_name, attrs in sensitive_names_map.items():
        for attr in attrs:
            if name == attr:
                return group_name.value

    # Check startswith / endswith
    for group_name, attrs in sensitive_names_map.items():
        for attr in attrs:
            if name.startswith(attr) or name.endswith(attr):
                return group_name.value

    # Check distance < threshold
    for group_name, attrs in sensitive_names_map.items():
        for attr in attrs:
            dist = str_distance(name, attr)
            if dist < threshold:
                return group_name.value

    return None


def _detect_names_dict(
    names: List[str], threshold: float = 0.1, str_distance: Callable[[Optional[str], Optional[str]], float] = None
) -> Dict[str, Optional[str]]:
    """Creates a dictionary which maps the attribute names to the corresponding sensitive attribute.

    Args:
        names (List[str]):
            List of attribute names.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.

    Returns:
        Dict[str, str]:
            A dictionary containing a mapping from attribute names to a string representing the corresponding
            sensitive attribute category or None.

    Examples:
        >>> _detect_names_dict(["age", "gender", "legality", "risk"])
        {"age": "Age", "gender": "Gender", "legality": None, "risk": None}
    """

    names_dict = dict()

    for name in names:
        names_dict[name] = _detect_name(name, threshold=threshold, str_distance=str_distance)

    # Remove columns with 'None' values.
    for key, value in dict(names_dict).items():
        if value is None:
            del names_dict[key]

    return names_dict


def _deep_search(
    s: pd.Series, threshold: float = 0.1, str_distance: Callable[[Optional[str], Optional[str]], float] = None
) -> Optional[str]:

    # Avoid checking number values as they can be inconclusive.
    if s.dtype.kind in ["i", "f", "m", "M"]:
        return None

    str_distance = str_distance or _ro_distance

    # Coarse grain search to check if there is an exact match to avoid mismatches.
    for group_name, values in sensitive_values_map.items():
        # Skip sensitive groups that do not have defined possible values.
        if not values:
            continue
        pattern = "|".join(values)
        if s.isin(values).mean() > 0.2:
            return group_name.value

    for group_name, values in sensitive_values_map.items():
        if not values:
            continue
        pattern = "|".join(values)
        if s.str.contains(pattern).mean() > 0.6:
            return group_name.value

    # Fine grain search that will catch edge cases.
    for group_name, values in sensitive_values_map.items():
        for value in values:
            if s.map(lambda x: str_distance(x, value) < threshold).mean() > 0.1:
                return group_name.value
    return None
