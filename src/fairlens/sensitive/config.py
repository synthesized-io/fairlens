import json
import os
import pathlib
from typing import Dict, List, Union

dict_path = "./configs/config_engb.json"
attr_synonym_dict: Dict[str, List[str]] = {}
attr_value_dict: Dict[str, List[str]] = {}


def default_config():
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

    global attr_synonym_dict
    attr_synonym_dict = syn_dict
    global attr_value_dict
    attr_value_dict = val_dict


def change_config(config_path: Union[str, pathlib.Path]):
    PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(PROJ_DIR, config_path)

    json_file = open(json_path)
    config_dict = json.load(json_file)
    json_file.close()

    syn_dict = dict()
    val_dict = dict()

    for key, value in config_dict.items():
        syn_dict[key] = value["synonyms"]
        val_dict[key] = value["values"]

    global attr_synonym_dict
    attr_synonym_dict = syn_dict
    global attr_value_dict
    attr_value_dict = val_dict
