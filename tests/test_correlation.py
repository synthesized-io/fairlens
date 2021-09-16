import pandas as pd
import pytest

from fairlens import utils
from fairlens.metrics.correlation import (
    cramers_v,
    distance_cn_correlation,
    distance_nn_correlation,
    kruskal_wallis,
    pearson,
)
from fairlens.metrics.unified import correlation_matrix
from fairlens.sensitive.correlation import find_column_correlation, find_sensitive_correlations

pair_race = "race", "Ethnicity"
pair_age = "age", "Age"
pair_marital = "marital", "Family Status"
pair_gender = "gender", "Gender"
pair_nationality = "nationality", "Nationality"

epsilon = 1e-6


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
    res = {"score": [pair_gender]}
    assert find_sensitive_correlations(df) == res


def test_double_correlation():
    col_names = ["gender", "nationality", "random", "corr1", "corr2"]
    data = [
        ["woman", "spanish", 715, 10, 20],
        ["man", "spanish", 1008, 20, 20],
        ["man", "french", 932, 20, 10],
        ["woman", "french", 1300, 10, 10],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {"corr1": [pair_gender], "corr2": [pair_nationality]}
    assert find_sensitive_correlations(df) == res


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
    res = {"corr1": [pair_race, pair_marital]}
    assert find_sensitive_correlations(df, corr_cutoff=0.9) == res


def test_common_correlation():
    col_names = ["race", "age", "score", "entries", "marital", "credit", "corr1", "corr2"]
    data = [
        ["arabian", 21, 10, 2000, "married", 10, 60, 120],
        ["carribean", 20, 10, 3000, "single", 10, 90, 130],
        ["indo-european", 41, 10, 1900, "widowed", 10, 120, 210],
        ["carribean", 40, 10, 2000, "single", 10, 90, 220],
        ["indo-european", 42, 10, 2500, "widowed", 10, 120, 200],
        ["arabian", 19, 10, 2200, "married", 10, 60, 115],
        ["arabian", 21, 10, 2000, "married", 10, 60, 120],
        ["carribean", 20, 10, 3000, "single", 10, 90, 130],
        ["indo-european", 41, 10, 1900, "widowed", 10, 120, 210],
        ["carribean", 40, 10, 2000, "single", 10, 90, 220],
        ["indo-european", 42, 10, 2500, "widowed", 10, 120, 200],
        ["arabian", 19, 10, 2200, "married", 10, 60, 115],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res = {
        "corr1": [pair_race, pair_age, pair_marital],
        "corr2": [pair_race, pair_age, pair_marital],
        "entries": [pair_age],
    }
    assert find_sensitive_correlations(df) == res


def test_column_correlation():
    col_names = ["gender", "nationality", "random", "corr1", "corr2"]
    data = [
        ["woman", "spanish", 715, 10, 20],
        ["man", "spanish", 1008, 20, 20],
        ["man", "french", 932, 20, 10],
        ["woman", "french", 1300, 10, 10],
    ]
    df = pd.DataFrame(data, columns=col_names)
    res1 = [pair_gender]
    res2 = [pair_nationality]
    assert find_column_correlation("corr1", df) == res1
    assert find_column_correlation("corr2", df) == res2


def test_series_correlation():
    col_names = ["race", "age", "score", "entries", "marital", "credit"]
    data = [
        ["arabian", 21, 10, 2000, "married", 10],
        ["carribean", 20, 10, 3000, "single", 10],
        ["indo-european", 41, 10, 1900, "widowed", 10],
        ["carribean", 40, 10, 2000, "single", 10],
        ["indo-european", 42, 10, 2500, "widowed", 10],
        ["arabian", 19, 10, 2200, "married", 10],
        ["arabian", 21, 10, 2000, "married", 10],
        ["carribean", 20, 10, 3000, "single", 10],
        ["indo-european", 41, 10, 1900, "widowed", 10],
        ["carribean", 40, 10, 2000, "single", 10],
        ["indo-european", 42, 10, 2500, "widowed", 10],
        ["arabian", 19, 10, 2200, "married", 10],
    ]
    df = pd.DataFrame(data, columns=col_names)
    s1 = pd.Series([60, 90, 120, 90, 120, 60, 60, 90, 120, 90, 120, 60])
    s2 = pd.Series([120, 130, 210, 220, 200, 115, 120, 130, 210, 220, 200, 115])
    res1 = [pair_age, pair_race, pair_marital]
    res2 = [pair_age, pair_race, pair_marital]
    assert set(find_column_correlation(s1, df)) == set(res1)
    assert set(find_column_correlation(s2, df)) == set(res2)


def test_basic_nn_distance_corr():
    sr_a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sr_b = pd.Series([30.0, 10.0, 20.0, 60.0, 50.0, 40.0])

    assert distance_nn_correlation(sr_a, sr_b) > 0.75


def test_cn_basic_distance_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B"])
    sr_b = pd.Series([15, 45, 14, 16, 44, 46])

    assert distance_cn_correlation(sr_a, sr_b) > 0.8


def test_nn_unequal_series_corr():
    sr_a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sr_b = pd.Series([10.0, 20.0, 60.0])

    assert distance_nn_correlation(sr_a, sr_b) > 0.7


def test_cn_unequal_series_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E", "F", "F", "F", "F"])
    sr_b = pd.Series([100, 200, 99, 101, 201, 199, 299, 300, 301, 500, 501, 505, 10, 12, 1001, 1050])

    assert distance_cn_correlation(sr_a, sr_b) > 0.7


@pytest.mark.parametrize("dataset", ["titanic", "german_credit_data"])
def test_correlation_matrix(dataset):
    df = pd.read_csv(f"datasets/{dataset}.csv")
    num_num_metric = pearson
    cat_num_metric = kruskal_wallis
    cat_cat_metric = cramers_v

    matrix = correlation_matrix(
        df, num_num_metric=num_num_metric, cat_num_metric=cat_num_metric, cat_cat_metric=cat_cat_metric
    ).to_numpy()

    for i, r in enumerate(df.columns):
        for j, c in enumerate(df.columns):
            sr_a = utils.infer_dtype(df[r])
            sr_b = utils.infer_dtype(df[c])
            a_type = utils.infer_distr_type(sr_a)
            b_type = utils.infer_distr_type(sr_b)

            d = pd.DataFrame({"a": sr_a, "b": sr_b}).dropna().reset_index()

            if a_type.is_continuous() and b_type.is_continuous():
                corr = num_num_metric(d["a"], d["b"])

            elif b_type.is_continuous():
                corr = cat_num_metric(d["a"], d["b"])

            elif a_type.is_continuous():
                corr = cat_num_metric(d["b"], d["a"])

            else:
                corr = cat_cat_metric(d["a"], d["b"])

            assert matrix[i][j] - corr < epsilon
