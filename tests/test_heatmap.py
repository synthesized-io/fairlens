import pandas as pd

from fairlens.scorer import heatmap as hmp


def test_basic_nn_distance_corr():
    sr_a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sr_b = pd.Series([30.0, 10.0, 20.0, 60.0, 50.0, 40.0])

    assert hmp._distance_nn_correlation(sr_a, sr_b) > 0.75


def test_cn_basic_distance_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B"])
    sr_b = pd.Series([15, 45, 14, 16, 44, 46])

    assert hmp._distance_cn_correlation(sr_a, sr_b) > 0.8


def test_nn_unequal_series_corr():
    sr_a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sr_b = pd.Series([10.0, 20.0, 60.0])

    assert hmp._distance_nn_correlation(sr_a, sr_b) > 0.7


def test_cn_unequal_series_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E", "F", "F", "F", "F"])
    sr_b = pd.Series([100, 200, 99, 101, 201, 199, 299, 300, 301, 500, 501, 505, 10, 12, 1001, 1050])

    assert hmp._distance_cn_correlation(sr_a, sr_b) > 0.7
