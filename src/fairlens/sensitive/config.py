import json
import os

dict_path = "./configs/config_engb.json"


def default_dict():
    PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(PROJ_DIR, dict_path)

    json_file = open(json_path)
    config_dict = json.load(json_file)
    json_file.close()

    print(config_dict)


if __name__ == "__main__":
    default_dict()
