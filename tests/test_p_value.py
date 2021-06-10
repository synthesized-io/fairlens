import numpy as np
import pandas as pd

from fairlens.bias import p_value as pv
from fairlens.bias.metrics import stat_distance

# from fairlens.bias.metrics import stat_distance

epsilon = 1e-5
df = pd.read_csv("datasets/compas.csv")
target_attr = "RawScore"
group1 = {"Ethnicity": ["Caucasian"]}
group2 = {"Ethnicity": ["African-American"]}


def test_binomial():
    assert abs(pv.binominal_proportion_p_value(0.2, 0.1, 10) - (1 - (0.9 ** 10 + 0.9 ** 9))) < epsilon
    assert stat_distance(df, "", pd.Series([1, 1]), pd.Series([0, 0]), p_value=True)[1] == 0
    assert stat_distance(df, "", pd.Series([1, 0]), pd.Series([1, 0]), p_value=True)[1] == 1
    assert stat_distance(df, "", pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 1, 0]), p_value=True)[1] == 0.625


def test_bootstrapping():
    pass


def test_permutation():
    assert stat_distance(df, "", pd.Series(np.ones(100)), pd.Series(np.zeros(100)), mode="emd", p_value=True)[1] == 0
    assert stat_distance(df, "", pd.Series([1, 0]), pd.Series([1, 0]), mode="emd", p_value=True)[1] == 1

    print(stat_distance(df, target_attr, group1, group2, mode="emd", p_value=True))
    assert stat_distance(df, target_attr, group1, group2, mode="emd", p_value=True)[1] == 0


def test_ks_distance():
    pass
