import pandas as pd

from fairlens.sensitive import detection as dt


def test_detect_name():
    assert dt._detect_name("Creed") == "Religion"
    assert dt._detect_name("date birth of", threshold=0.1) is None
    assert dt._detect_name("date birth of", threshold=0.5) == "Age"
    assert dt._detect_name("Sexual Preference") == "Gender"


def test_detect_names():
    cols = ["age", "gender", "legality", "risk"]
    assert dt.detect_names(cols) == ["age", "gender"]


def test_detect_names_dict():
    cols = ["age", "gender", "legality", "risk"]
    res = {"age": "Age", "gender": "Gender"}
    assert dt.detect_names_dict(cols) == res


def test_detect_names_dataframe_simple():
    col_names = ["age", "sexual orientation", "salary", "score"]
    df = pd.DataFrame(columns=col_names)
    res = ["age", "sexual orientation"]
    assert dt.detect_names_dataframe(df) == res


def test_detect_names_dataframe_dict_simple():
    col_names = ["native", "location", "house", "subscription", "salary", "religion", "score"]
    df = pd.DataFrame(columns=col_names)
    res = {
        "native": "Nationality",
        "location": "Nationality",
        "house": "Family Status",
        "religion": "Religion",
    }
    assert dt.detect_names_dict_dataframe(df) == res


def test_detect_names_dataframe_deep():
    col_names = ["A", "B", "C", "Salary", "D", "Score"]
    data = [
        ["male", "hearing impairment", "heterosexual", "50000", "christianity", "10"],
        ["female", "obsessive compulsive disorder", "asexual", "60000", "daoism", "10"],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = ["A", "B", "C", "D"]
    assert set(dt.detect_names_dataframe(df, deep_search=True)) == set(res)


def test_detect_names_dict_dataframe_deep():
    col_names = ["Rand", "A", "B", "Score", "Credit", "xyzqwe", "D"]
    data = [
        ["scottish", "asian", "male", "5", "80", "no religion", "sight loss"],
        ["romanian", "caucasian", "female", "4", "100", "christianity", "arthritis"],
        ["bulgarian", "european", "agender", "10", "40", "islam", "sclerosis"],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"Rand": "Nationality", "A": "Ethnicity", "B": "Gender", "xyzqwe": "Religion", "D": "Disability"}
    assert dt.detect_names_dict_dataframe(df, deep_search=True) == res


def test_dataframe_names_stress():
    col_names = ["xyz", "abc", "A", "B", "C", "D"]
    data = [
        [None, "romanian", "80", "100", None, "sclerosis"],
        ["asian", None, None, "200", "islam", "sight loss"],
        [None, None, "100", "150", "christian", None],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = ["xyz", "abc", "C", "D"]
    assert set(dt.detect_names_dataframe(df, deep_search=True)) == set(res)


def test_dataframe_dict_stress():
    col_names = ["xyz", "abc", "A", "B", "C", "D"]
    data = [
        [None, "romanian", "80", "100", None, "sclerosis"],
        ["asian", None, None, "200", "islam", "sight loss"],
        [None, None, "100", "150", "christian", None],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"xyz": "Ethnicity", "abc": "Nationality", "C": "Religion", "D": "Disability"}
    assert dt.detect_names_dict_dataframe(df, deep_search=True) == res
