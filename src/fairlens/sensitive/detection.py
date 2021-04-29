from difflib import SequenceMatcher
from enum import Enum
from typing import Callable, Dict, List, Optional

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


def _ro_distance(s1: str, s2: str) -> float:
    """Computes a distance between the input strings using the Ratcliff-Obershelp algorithm."""
    if s1 is None or s2 is None:
        return 1

    return 1 - SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _detect_name(name: str, threshold: float = 0.1, str_distance: Callable[[str, str], float] = None) -> Optional[str]:
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


def detect_names(
    names: List[str], threshold: float = 0.1, str_distance: Callable[[str, str], float] = None
) -> List[str]:
    """Filters the sensitive attributes in a list.

    Args:
        names (List[str]):
            List of attribute names.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.

    Returns:
        List[str]:
            List containing the sensitive attribute names.

    Examples:
        >>> detect_names(["age", "gender", "legality", "risk"])
        ["age", "gender"]
    """

    sensitive_attrs = []

    for name in names:
        if _detect_name(name, threshold=threshold, str_distance=str_distance):
            sensitive_attrs.append(name)

    return sensitive_attrs


def detect_names_dataframe(
    df: pd.DataFrame,
    threshold: float = 0.1,
    str_distance: Callable[[str, str], float] = None,
    deep_search: bool = False,
) -> List[str]:
    """[summary]

    Args:
        df (pd.DataFrame):
            Pandas dataframe that will be analysed.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.

    Returns:
        List[str]:
            List containing the sensitive attribute names.
    """
    cols = df.columns
    sensitive_cols = detect_names(cols, threshold, str_distance)

    str_distance = str_distance or _ro_distance

    if deep_search:
        non_sensitive_cols = list(set(cols) - set(sensitive_cols))

        for non_sensitive_col in non_sensitive_cols:
            # Avoid checking number values as they can be inconclusive.
            if df[non_sensitive_col].dtype.kind in ["i", "f", "m", "M"]:
                continue
            for _, values in sensitive_values_map.items():
                for value in values:
                    if (
                        df[non_sensitive_col].str.contains(value).any()
                        or df[non_sensitive_col].str.startswith(value).any()
                        or df[non_sensitive_col].str.endswith(value).any()
                        or df[non_sensitive_col].map(lambda x: str_distance(x, value) < threshold).any()
                    ):

                        sensitive_cols.append(non_sensitive_col)
        return sensitive_cols
    else:
        return sensitive_cols


def detect_names_dict(
    names: List[str], threshold: float = 0.1, str_distance: Callable[[str, str], float] = None
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
            sensitive attribute or None.

    Examples:
        >>> detect_names_dict(["age", "gender", "legality", "risk"])
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


def detect_names_dict_dataframe(
    df: pd.DataFrame,
    threshold: float = 0.1,
    str_distance: Callable[[str, str], float] = None,
    deep_search: bool = False,
) -> Dict[str, Optional[str]]:
    """[summary]

    Args:
        df (pd.DataFrame):
            Pandas dataframe that will be analysed.
        threshold (float, optional):
            The threshold for the string distance function. Defaults to 0.1.
        str_distance (Callable[[str, str], float], optional):
            The string distance function. Defaults to Ratcliff-Obershelp algorithm.

    Returns:
        Dict[str, Optional[str]]:
            A dictionary containing a mapping from attribute names to a string representing the corresponding
            sensitive attribute or None.
    """
    cols = df.columns
    sensitive_dict = detect_names_dict(cols, threshold, str_distance)
    sensitive_cols = list(sensitive_dict.keys())

    str_distance = str_distance or _ro_distance

    if deep_search:
        non_sensitive_cols = list(set(cols) - set(sensitive_cols))

        for non_sensitive_col in non_sensitive_cols:
            # Avoid checking number values as they can be inconclusive.
            if df[non_sensitive_col].dtype.kind in ["i", "f", "m", "M"]:
                continue
            for group_name, values in sensitive_values_map.items():
                for value in values:
                    if (
                        df[non_sensitive_col].str.contains(value).any()
                        or df[non_sensitive_col].str.startswith(value).any()
                        or df[non_sensitive_col].str.endswith(value).any()
                        or df[non_sensitive_col].map(lambda x: str_distance(x, value) < threshold).any()
                    ):

                        sensitive_dict[non_sensitive_col] = group_name.value
        return sensitive_dict
    else:
        return sensitive_dict
