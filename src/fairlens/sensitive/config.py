import json
import os
import pathlib
from typing import Any, Tuple, Union

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(PROJ_DIR, "./configs/config_engb.json")


def load_config(config_path: Union[str, pathlib.Path] = DEFAULT_PATH) -> Tuple[Any, Any]:
    with open(config_path) as json_file:
        config_dict = json.load(json_file)

    return config_dict["synonyms"], config_dict["values"]
