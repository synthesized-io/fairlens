import os

import pandas as pd

from fairlens.sensitive import detection as dt

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_CONFIG_PATH = os.path.join(TEST_DIR, "../src/fairlens/sensitive/configs/config_mock.json")
ENGB_CONFIG_PATH = os.path.join(TEST_DIR, "../src/fairlens/sensitive/configs/config_engb.json")

dfc = pd.read_csv("datasets/compas.csv")


def test_detect_name():
    assert dt._detect_name("Creed") == "Religion"
    assert dt._detect_name("date birth of", threshold=0.1) is None
    assert dt._detect_name("date birth of", threshold=0.5) == "Age"
    assert dt._detect_name("Sexual Preference") == "Sexual Orientation"


def test_detect_names():
    cols = ["age", "gender", "legality", "risk"]
    assert list(dt.detect_names_df(cols).keys()) == ["age", "gender"]


def test_detect_names_dict():
    cols = ["age", "gender", "legality", "risk"]
    res = {"age": "Age", "gender": "Gender"}
    assert dt.detect_names_df(cols) == res


def test_detect_names_dataframe_simple():
    col_names = ["age", "sexual orientation", "salary", "score"]
    df = pd.DataFrame(columns=col_names)
    res = ["age", "sexual orientation"]
    assert list(dt.detect_names_df(df).keys()) == res


def test_detect_names_dataframe_dict_simple():
    col_names = ["native", "location", "house", "subscription", "salary", "religion", "score"]
    df = pd.DataFrame(columns=col_names)
    res = {
        "native": "Nationality",
        "location": "Nationality",
        "house": "Family Status",
        "religion": "Religion",
    }
    assert dt.detect_names_df(df) == res


def test_detect_names_dataframe_deep():
    col_names = ["A", "B", "C", "Salary", "D", "Score"]
    data = [
        ["male", "hearing impairment", "heterosexual", "50000", "christianity", "10"],
        ["female", "obsessive compulsive disorder", "asexual", "60000", "daoism", "10"],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = ["A", "B", "C", "D"]
    assert set(dt.detect_names_df(df, deep_search=True).keys()) == set(res)


def test_detect_names_dict_dataframe_deep():
    col_names = ["Rand", "A", "B", "Score", "Credit", "xyzqwe", "D"]
    data = [
        ["scottish", "asian", "male", "5", "80", "no religion", "sight loss"],
        ["romanian", "caucasian", "female", "4", "100", "christianity", "arthritis"],
        ["bulgarian", "european", "agender", "10", "40", "islam", "sclerosis"],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"Rand": "Nationality", "A": "Ethnicity", "B": "Gender", "xyzqwe": "Religion", "D": "Disability"}
    assert dt.detect_names_df(df, threshold=0.01, deep_search=True) == res


def test_dataframe_names_stress():
    col_names = ["xyz", "abc", "A", "B", "C", "D"]
    data = [
        [None, "romanian", "80", "100", None, "sclerosis"],
        ["asian", None, None, "200", "islam", "sight loss"],
        [None, None, "100", "150", "christian", None],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = ["xyz", "abc", "C", "D"]
    assert set(dt.detect_names_df(df, deep_search=True).keys()) == set(res)


def test_dataframe_dict_stress():
    col_names = ["xyz", "abc", "A", "B", "C", "D"]
    data = [
        [None, "scottish", "80", "100", None, "sclerosis"],
        ["asian", None, None, "200", "islam", "sight loss"],
        [None, None, "100", "150", "christian", None],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"xyz": "Ethnicity", "abc": "Nationality", "C": "Religion", "D": "Disability"}
    assert dt.detect_names_df(df, deep_search=True) == res


def test_dataframe_names_numbers():
    col_names = ["A", "B", "C"]
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    df = pd.DataFrame(data, columns=col_names)
    res = []
    assert list(dt.detect_names_df(df, deep_search=True).keys()) == res


def test_dataframe_dict_numbers():
    col_names = ["X", "Y", "Z"]
    data = [[1, 1, 2], [3, 5, 8], [13, 21, 34]]
    df = pd.DataFrame(data, columns=col_names)
    res = {}
    assert dt.detect_names_df(df, deep_search=True) == res


def test_compas_detect_shallow():
    res = {
        "DateOfBirth": "Age",
        "Ethnicity": "Ethnicity",
        "Language": "Nationality",
        "MaritalStatus": "Family Status",
        "Sex": "Gender",
    }
    assert dt.detect_names_df(dfc) == res


def test_compas_detect_deep():
    res = {
        "DateOfBirth": "Age",
        "Ethnicity": "Ethnicity",
        "Language": "Nationality",
        "MaritalStatus": "Family Status",
        "Sex": "Gender",
    }
    assert dt.detect_names_df(dfc, deep_search=True) == res


def test_default_config():
    col_names = [
        "gender",
        "religion",
        "nationality",
        "family status",
        "sexual orientation",
        "age",
        "ethnicity",
        "disability",
    ]
    df = pd.DataFrame(columns=col_names)
    res = {
        "gender": "Gender",
        "religion": "Religion",
        "nationality": "Nationality",
        "family status": "Family Status",
        "sexual orientation": "Sexual Orientation",
        "age": "Age",
        "ethnicity": "Ethnicity",
        "disability": "Disability",
    }
    assert dt.detect_names_df(df) == res


def test_change_config_shallow():
    col_names = ["Credit", "mammal", "reptile", "Score", "Gender"]
    data = [[1, "dog", None, 10, "a"], [2, None, "lizard", 12, "b"], [3, "cat", None, 10, "c"]]
    df = pd.DataFrame(data, columns=col_names)
    res = {"mammal": "Mammals", "reptile": "Reptiles"}
    assert dt.detect_names_df(df, config_path=MOCK_CONFIG_PATH) == res


def test_change_config_deep():
    col_names = ["Credit", "M", "R", "Score", "Gender"]
    data = [[1, "dog", None, 10, "a"], [2, None, "yellow chameleon", 12, "b"], [3, "cat", None, 10, "c"]]
    df = pd.DataFrame(data, columns=col_names)
    res = {"M": "Mammals", "R": "Reptiles"}
    assert dt.detect_names_df(df, deep_search=True, config_path=MOCK_CONFIG_PATH) == res


def test_double_config_shallow():
    col_names = ["Credit", "mammal", "reptile", "Score", "gender", "Rand", "ethnicity"]
    df = pd.DataFrame(columns=col_names)
    res1 = {"gender": "Gender", "ethnicity": "Ethnicity"}
    res2 = {"mammal": "Mammals", "reptile": "Reptiles"}
    assert dt.detect_names_df(df, config_path=ENGB_CONFIG_PATH) == res1
    assert dt.detect_names_df(df, config_path=MOCK_CONFIG_PATH) == res2


def test_double_config_deep():
    col_names = ["Credit", "B", "R", "Score", "F1", "Rand", "F2"]
    data = [
        [10, "golden oriole", "islam", 20, "white shark", "xyz", "married"],
        [20, "osprey", "christian", 10, "angler fish", "abc", "divorced"],
        [30, None, None, 30, None, None, None],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res1 = {"R": "Religion", "F2": "Family Status"}
    res2 = {"B": "Birds", "F1": "Fish"}
    assert dt.detect_names_df(df, deep_search=True, config_path=ENGB_CONFIG_PATH) == res1
    assert dt.detect_names_df(df, deep_search=True, config_path=MOCK_CONFIG_PATH) == res2


if __name__ == "__main__":
    dfc = pd.read_csv("datasets/compas.csv")
    res = dt.detect_names_df(dfc, deep_search=True)
    print(res)
