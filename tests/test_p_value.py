import numpy as np
import pandas as pd

from fairlens.bias.metrics import MeanDistance, stat_distance
from fairlens.bias.p_value import binominal_proportion_p_value as bin_prop
from fairlens.bias.p_value import bootstrap_binned_statistic as bootstrap_binned
from fairlens.bias.p_value import bootstrap_statistic as bootstrap
from fairlens.bias.p_value import permutation_statistic as perm_stat
from fairlens.bias.p_value import resampling_interval, resampling_p_value

epsilon = 1e-5
df = pd.read_csv("datasets/compas.csv")
target_attr = "RawScore"
group1 = {"Ethnicity": ["Caucasian"]}
group2 = {"Ethnicity": ["African-American"]}


def test_binomial():
    assert abs(bin_prop(0.2, 0.1, 10) - (1 - (0.9 ** 10 + 0.9 ** 9))) < epsilon
    assert stat_distance(df, "", pd.Series([1, 1]), pd.Series([0, 0]), p_value=True)[1] == 0
    assert stat_distance(df, "", pd.Series([1, 0]), pd.Series([1, 0]), p_value=True)[1] == 1
    assert stat_distance(df, "", pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 1, 0]), p_value=True)[1] == 0.625


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
