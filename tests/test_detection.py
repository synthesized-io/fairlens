import pandas as pd

from fairlens.sensitive import detection as dt


def test_detect_name():
    assert dt._detect_name("Creed") == "Religion"
    assert dt._detect_name("date birth of", threshold=0.1) is None
    assert dt._detect_name("date birth of", threshold=0.5) == "Age"
    assert dt._detect_name("Sexual Preference") == "Gender"


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


def test_correlation():
    col_names = ["gender", "random", "score"]
    data = [["male", 10, 60], ["female", 10, 80], ["male", 10, 60], ["female", 10, 80]]
    df = pd.DataFrame(data, columns=col_names)
    res = {"score": [("gender", "Gender")]}
    assert dt.find_sensitive_correlations(df) == res


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
    assert dt.find_sensitive_correlations(df) == res


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
    assert dt.find_sensitive_correlations(df, corr_cutoff=0.9) == res


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
    assert dt.find_sensitive_correlations(df) == res


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
    assert dt.find_column_correlation("corr1", df) == res1
    assert dt.find_column_correlation("corr2", df) == res2


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
    assert set(dt.find_column_correlation(s1, df, corr_cutoff=0.9)) == set(res1)
    assert set(dt.find_column_correlation(s2, df, corr_cutoff=0.9)) == set(res2)
