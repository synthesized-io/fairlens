import numpy as np
import pandas as pd
import pytest

from fairlens.metrics.distance import BinomialDistance, MeanDistance
from fairlens.metrics.significance import binominal_proportion_p_value as bin_prop
from fairlens.metrics.significance import bootstrap_binned_statistic as bootstrap_binned
from fairlens.metrics.significance import bootstrap_statistic as bootstrap
from fairlens.metrics.significance import brunner_munzel_test as bm_test
from fairlens.metrics.significance import permutation_statistic as perm_stat
from fairlens.metrics.significance import resampling_interval, resampling_p_value

epsilon = 1e-5
df = pd.read_csv("datasets/compas.csv")
target_attr = "RawScore"
group1 = {"Ethnicity": ["Caucasian"]}
group2 = {"Ethnicity": ["African-American"]}


def test_binomial():
    assert abs(bin_prop(0.2, 0.1, 10) - (1 - (0.9**10 + 0.9**9))) < epsilon
    assert BinomialDistance().p_value(pd.Series([1, 1]), pd.Series([0, 0])) == 0
    assert BinomialDistance().p_value(pd.Series([1, 0]), pd.Series([1, 0])) == 1
    assert BinomialDistance().p_value(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 1, 0])) == 0.625


def test_bootstrap():
    assert bootstrap(pd.Series([1]), pd.Series([0]), MeanDistance().distance, 100).min() == 1
    assert bootstrap(pd.Series(range(2)), pd.Series(range(2, 4)), MeanDistance().distance, 1000).max() == 3


def test_bootstrap_binned():
    def distance(h_x, h_y):
        return np.linalg.norm(h_x - h_y, ord=1)

    assert bootstrap_binned(pd.Series([1, 3, 0]), pd.Series([1, 4, 3]), distance, 10000).max() == 12


def test_permutation():
    assert perm_stat(pd.Series([1]), pd.Series([0]), MeanDistance().distance, 100).min() == 1
    assert perm_stat(pd.Series([1, 1]), pd.Series([0, 0]), MeanDistance().distance, 1000).min() == 0
    assert perm_stat(pd.Series(range(5)), pd.Series(range(5, 10)), MeanDistance().distance, 1000).max() == 5


def test_resampled_pvalue():
    assert resampling_p_value(12, pd.Series([13, 11]), "two-sided") == 0.5
    assert resampling_p_value(12, pd.Series([13, 11]), "greater") == 0.5
    assert resampling_p_value(12, pd.Series([13, 11]), "less") == 0.5

    assert resampling_p_value(12, pd.Series([15, 14, 13, 11]), "two-sided") == 0.75
    assert resampling_p_value(12, pd.Series([15, 14, 13, 11]), "greater") == 0.75
    assert resampling_p_value(12, pd.Series([15, 14, 13, 11]), "less") == 0.25

    assert resampling_p_value(0, pd.Series([-2, -1, 0, 1]), "two-sided") == 1
    assert resampling_p_value(0, pd.Series([-2, -1, 0, 1]), "greater") == 0.5
    assert resampling_p_value(0, pd.Series([-2, -1, 0, 1]), "less") == 0.5


def test_resampled_interval():
    assert resampling_interval(3, pd.Series([1, 4, 2, 3, 5]), cl=0.5) == (2.0, 4.0)
    assert resampling_interval(50, pd.Series(np.arange(101)), cl=0.8) == (10, 90)


def test_brunner_munzel():
    data = [
        [10, "Caucasian", "24/4/1999", "Rejected"],
        [25, "African-American", "19/7/1997", "Accepted"],
        [15, "Hispanic", "31/12/2001", "Accepted"],
        [34, "Other", "20/2/1998", "Rejected"],
        [35, "Caucasian", "2/3/2002", "Accepted"],
        [56, "Hispanic", "6/6/1997", "Accepted"],
        [80, "African-American", "4/5/2000", "Accepted"],
        [100, "African-American", "3/1/1996", "Accepted"],
        [134, "Caucasian", "24/4/1999", "Rejected"],
        [21, "African-American", "19/7/1997", "Rejected"],
        [14, "Hispanic", "31/12/2001", "Rejected"],
        [98, "Other", "20/2/1998", "Rejected"],
        [76, "Caucasian", "2/3/2002", "Accepted"],
        [51, "Hispanic", "6/6/1997", "Accepted"],
        [82, "African-American", "4/5/2000", "Rejected"],
        [145, "African-American", "3/1/1996", "Accepted"],
    ]
    df = pd.DataFrame(data=data, columns=["score", "race", "date", "status"])
    group1 = {"race": ["African-American"]}
    group2 = {"race": ["Caucasian"]}
    res = bm_test(df, target_attr="score", group1=group1, group2=group2)
    assert res[0] == pytest.approx(-0.5883, rel=1e-3)
    assert res[1] == pytest.approx(0.5777, rel=1e-3)
