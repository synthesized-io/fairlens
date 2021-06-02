import json
import os
import pathlib
from typing import Dict, List, Union

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(PROJ_DIR, "./configs/config_engb.json")

attr_synonym_dict: Dict[str, List[str]] = {}
attr_value_dict: Dict[str, List[str]] = {}


def load_config(config_path: Union[str, pathlib.Path] = DEFAULT_PATH):
    with open(config_path) as json_file:
        config_dict = json.load(json_file)

    global attr_synonym_dict
    attr_synonym_dict = config_dict["synonyms"]
    global attr_value_dict
    attr_value_dict = config_dict["values"]
