import pandas as pd

from fairlens.bias import metrics
from fairlens.bias.metrics import stat_distance

df = pd.read_csv("datasets/compas.csv")
pred1 = df["Ethnicity"] == "Caucasian"
pred2 = (df["Ethnicity"] == "African-American") | (df["Ethnicity"] == "African-Am")
target_attr = "RawScore"
group1 = df[pred1][target_attr]
group2 = df[pred2][target_attr]
target = df[target_attr]


def test_stat_distance():
    x = {"Ethnicity": ["Caucasian"]}
    y = {"Ethnicity": [e for e in list(df["Ethnicity"].unique()) if e != "Caucasian"]}
    xy = {"Ethnicity": list(df["Ethnicity"].unique())}

    res = metrics.EarthMoversDistance(target, group1, target).distance
    assert stat_distance(df, target_attr, group1, target, mode="emd") == res
    assert stat_distance(df, target_attr, x, xy, mode="emd") == res

    res = stat_distance(df, target_attr, x, mode="emd")
    assert stat_distance(df, target_attr, df[pred1][target_attr], df[~pred1][target_attr], mode="emd") == res
    assert stat_distance(df, target_attr, x, y, mode="emd") == res


def test_stat_distance_auto():
    res = stat_distance(df, target_attr, group1, group2, mode="auto")
    assert stat_distance(df, target_attr, group1, group2, mode="ks_distance") == res


def test_class_imbalance():
    assert metrics.ClassImbalance(target, group1, group1).distance == 0
    assert metrics.ClassImbalance(pd.Series([1, 2, 1]), pd.Series([1, 2]), pd.Series([1])).distance == 0.5
    assert metrics.ClassImbalance(pd.Series([1, 2, 3, 4]), pd.Series([1, 2]), pd.Series([3, 4])).distance == 0


def test_binomial_distance():
    assert metrics.BinomialDistance(target, group1, group1).distance == 0
    assert metrics.BinomialDistance(pd.Series([42]), pd.Series([1, 1]), pd.Series([0, 0])).distance == 1
    assert metrics.BinomialDistance(pd.Series([42]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert metrics.BinomialDistance(pd.Series([42]), pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])).distance == 0.5


def test_emd():
    assert metrics.EarthMoversDistance(target, group1, group1).distance == 0


def test_emd_categorical():
    assert metrics.EarthMoversDistanceCategorical(target, group1, group1).distance == 0


def test_ks_distance():
    assert metrics.KolmogorovSmirnovDistance(target, group1, group1).distance == 0


def test_kl_divergence():
    assert metrics.KullbackLeiblerDivergence(target, group1, group1).distance == 0


def test_js_divergence():
    assert metrics.JensenShannonDivergence(target, target, target).distance == 0


def test_norm():
    assert metrics.LNorm(target, group1, group1).distance == 0


def test_hellinger():
    assert metrics.HellingerDistance(target, group1, group1).distance == 0
