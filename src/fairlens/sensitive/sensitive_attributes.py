from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, List, Optional


class SensitiveNames(Enum):
    """
    This is a list of
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


def detect_name(name: str, threshold: float = 0.1) -> Optional[str]:
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


def detect_names(names: List[str], threshold: float = 0.1) -> List[str]:
    sensitive_attrs = []
    for name in names:
        if detect_name(name, threshold=threshold):
            sensitive_attrs.append(name)

    return sensitive_attrs


def detect_names_dict(names: List[str], threshold: float = 0.1) -> Dict[str, str]:
    names_dict = dict()
    for name in names:
        attr = detect_name(name, threshold=threshold)
        if attr is not None:
            names_dict[name] = attr

    return names_dict


def str_distance(s1: str, s2: str) -> float:
    return 1 - SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
