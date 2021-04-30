import json
import os
from typing import Dict, List

dict_path = "./configs/config_engb.json"
attr_synonym_dict: Dict[str, List[str]] = {}
attr_value_dict: Dict[str, List[str]] = {}

# @dataclass
# class SensitiveAttribute:
#     name: str
#     synonyms: List[str]
#     values: List[str]


def default_dict():
    PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(PROJ_DIR, dict_path)

    json_file = open(json_path)
    config_dict = json.load(json_file)
    json_file.close()

    syn_dict = dict()
    val_dict = dict()

    for key, value in config_dict.items():
        syn_dict[key] = value["synonyms"]
        val_dict[key] = value["values"]

    print(syn_dict)
    print(val_dict)


if __name__ == "__main__":
    default_dict()
