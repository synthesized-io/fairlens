from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class SensitiveNames(Enum):
    Age: str = "Age"
    Gender: str = "Gender"
    Ethnicity: str = "Ethnicity"
    Religion: str = "Religion"
    Nationality: str = "Nationality"
    FamilyStatus: str = "Family Status"
    Disability: str = "Disability"
    SexualOrientation: str = "Sexual Orientation"


class SensitiveNamesDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

        self.sensitive_names_map: Dict["SensitiveNames", List[str]] = {
            SensitiveNames.Age: ["age", "DOB", "birth", "youth", "elder", "senior"],
            SensitiveNames.Gender: ["gender", "sex"],
            SensitiveNames.Ethnicity: ["race", "color", "ethnic", "breed", "culture"],
            SensitiveNames.Nationality: ["nation", "geography", "location", "native", "country", "region"],
            SensitiveNames.Religion: ["religion", "creed", "cult", "doctrine"],
            SensitiveNames.FamilyStatus: ["family", "house", "marital", "children", "partner", "pregnant"],
            SensitiveNames.Disability: ["disability", "impairment"],
            SensitiveNames.SexualOrientation: ["sexual", "orientation", "attracted"],
        }

    def detect_name(self, name: str) -> Optional[Any]:
        name = name.lower()

        # Check exact match
        for group_name, attrs in self.sensitive_names_map.items():
            for attr in attrs:
                if name == attr:
                    return group_name.value

        # Check startswith / endswith
        for group_name, attrs in self.sensitive_names_map.items():
            for attr in attrs:
                if name.startswith(attr) or name.endswith(attr):
                    return group_name.value

        # Check distance < threshold
        for group_name, attrs in self.sensitive_names_map.items():
            for attr in attrs:
                dist = self.str_distance(name, attr)
                if dist < self.threshold:
                    return group_name.value
        return None

    def detect_names(self, names: List[str]) -> List[Optional[str]]:
        group_names = []
        for name in names:
            group_names.append(self.detect_name(name))

        return group_names

    def detect_names_dict(self, names: List[str]) -> Dict[str, str]:
        detected_names = self.detect_names(names)
        names_dict = dict()
        for detected_name, name in zip(detected_names, names):
            if detected_name is not None:
                names_dict[name] = detected_name
        return names_dict

    @staticmethod
    def str_distance(s1: str, s2: str) -> float:
        return 1 - SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def sensitive_attr_concat_name(sensitive_attr: Union[List[str], str]) -> str:
    if isinstance(sensitive_attr, list):
        if len(sensitive_attr) == 1:
            return sensitive_attr[0]
        else:
            return "({})".format(", ".join(sensitive_attr))
    elif isinstance(sensitive_attr, str):
        return sensitive_attr
    else:
        raise ValueError(sensitive_attr)
