import os

import pandas as pd

from fairlens.sensitive import correlation as corr
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
    dfc_deep = pd.read_csv("datasets/compas.csv")
    dfc_deep = dfc_deep.rename(columns={"Ethnicity": "A", "Language": "Random", "MaritalStatus": "B", "Sex": "C"})
    res = {
        "DateOfBirth": "Age",
        "A": "Ethnicity",
        "Random": "Nationality",
        "B": "Family Status",
        "C": "Gender",
    }
    assert dt.detect_names_df(dfc_deep, deep_search=True) == res


def test_correlation():
    col_names = ["gender", "random", "score"]
    data = [
        ["male", 10, 60],
        ["female", 10, 80],
        ["male", 10, 60],
        ["female", 10, 80],
        ["male", 9, 59],
        ["female", 11, 80],
        ["male", 12, 61],
        ["female", 10, 83],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"score": [("gender", "Gender")]}
    assert corr.find_sensitive_correlations(df) == res


def test_double_correlation():
    col_names = ["gender", "nationality", "random", "corr1", "corr2"]
    data = [
        ["woman", "spanish", 715, 10, 20],
        ["man", "spanish", 1008, 20, 20],
        ["man", "french", 932, 20, 10],
        ["woman", "french", 1300, 10, 10],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"corr1": [("gender", "Gender")], "corr2": [("nationality", "Nationality")]}
    assert corr.find_sensitive_correlations(df) == res


def test_multiple_correlation():
    col_names = ["race", "age", "score", "entries", "marital", "credit", "corr1"]
    data = [
        ["arabian", 21, 10, 2000, "married", 10, 60],
        ["carribean", 20, 10, 3000, "single", 10, 90],
        ["indo-european", 41, 10, 1900, "widowed", 10, 120],
        ["carribean", 40, 10, 2000, "single", 10, 90],
        ["indo-european", 42, 10, 2500, "widowed", 10, 120],
        ["arabian", 19, 10, 2200, "married", 10, 60],
    ]
    df = pd.DataFrame(data, columns=col_names)
    # The first series is correlated with the "race" and "family status" columns, while the second is
    # correlated with the "age" column
    res = {"corr1": [("race", "Ethnicity"), ("marital", "Family Status")]}
    assert corr.find_sensitive_correlations(df, corr_cutoff=0.9) == res


def test_common_correlation():
    col_names = ["race", "age", "score", "entries", "marital", "credit", "corr1", "corr2"]
    data = [
        ["arabian", 21, 10, 2000, "married", 10, 60, 120],
        ["carribean", 20, 10, 3000, "single", 10, 90, 130],
        ["indo-european", 41, 10, 1900, "widowed", 10, 120, 210],
        ["carribean", 40, 10, 2000, "single", 10, 90, 220],
        ["indo-european", 42, 10, 2500, "widowed", 10, 120, 200],
        ["arabian", 19, 10, 2200, "married", 10, 60, 115],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {
        "corr1": [("race", "Ethnicity"), ("age", "Age"), ("marital", "Family Status")],
        "corr2": [("race", "Ethnicity"), ("age", "Age"), ("marital", "Family Status")],
    }
    assert corr.find_sensitive_correlations(df) == res


def test_column_correlation():
    col_names = ["gender", "nationality", "random", "corr1", "corr2"]
    data = [
        ["woman", "spanish", 715, 10, 20],
        ["man", "spanish", 1008, 20, 20],
        ["man", "french", 932, 20, 10],
        ["woman", "french", 1300, 10, 10],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res1 = [("gender", "Gender")]
    res2 = [("nationality", "Nationality")]
    assert corr.find_column_correlation("corr1", df) == res1
    assert corr.find_column_correlation("corr2", df) == res2


def test_series_correlation():
    col_names = ["race", "age", "score", "entries", "marital", "credit"]
    data = [
        ["arabian", 21, 10, 2000, "married", 10],
        ["carribean", 20, 10, 3000, "single", 10],
        ["indo-european", 41, 10, 1900, "widowed", 10],
        ["carribean", 40, 10, 2000, "single", 10],
        ["indo-european", 42, 10, 2500, "widowed", 10],
        ["arabian", 19, 10, 2200, "married", 10],
    ]
    df = pd.DataFrame(data, columns=col_names)
    # The first series is correlated with the "race" and "family status" columns, while the second is
    # correlated with the "age" column
    s1 = pd.Series([60, 90, 120, 90, 120, 60])
    s2 = pd.Series([120, 130, 210, 220, 200, 115])
    res1 = [("race", "Ethnicity"), ("marital", "Family Status")]
    res2 = [("age", "Age")]
    assert set(corr.find_column_correlation(s1, df, corr_cutoff=0.9)) == set(res1)
    assert set(corr.find_column_correlation(s2, df, corr_cutoff=0.9)) == set(res2)


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
