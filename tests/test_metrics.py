import numpy as np
import pandas as pd

from fairlens.bias.metrics import (  # isort:skip
    BinomialDistance,
    ClassImbalance,
    EarthMoversDistance,
    EarthMoversDistanceCategorical,
    HellingerDistance,
    JensenShannonDivergence,
    KolmogorovSmirnovDistance,
    KullbackLeiblerDivergence,
    LNorm,
    stat_distance,
)

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

    res = EarthMoversDistance(target, group1, target).distance
    assert stat_distance(df, target_attr, group1, target, mode="emd") == res
    assert stat_distance(df, target_attr, x, xy, mode="emd") == res

    res = stat_distance(df, target_attr, x, mode="emd")
    assert stat_distance(df, target_attr, df[pred1][target_attr], df[~pred1][target_attr], mode="emd") == res
    assert stat_distance(df, target_attr, x, y, mode="emd") == res


def test_stat_distance_auto():
    res = stat_distance(df, target_attr, group1, group2, mode="auto")
    assert stat_distance(df, target_attr, group1, group2, mode="ks_distance") == res


def test_class_imbalance():
    assert ClassImbalance(target, group1, group1).distance == 0
    assert ClassImbalance(pd.Series([1, 2, 3, 4]), pd.Series([1, 2]), pd.Series([3, 4])).distance == 0
    assert ClassImbalance(pd.Series([1, 2, 1]), pd.Series([1, 2]), pd.Series([1])).distance == 0.5


def test_binomial_distance():
    assert BinomialDistance(target, group1, group1).distance == 0
    assert BinomialDistance(pd.Series([42]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert BinomialDistance(pd.Series([42]), pd.Series([1, 1]), pd.Series([0, 0])).distance == 1
    assert BinomialDistance(pd.Series([42]), pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])).distance == 0.5


def test_emd():
    assert EarthMoversDistance(target, group1, group1).distance == 0
    assert EarthMoversDistance(pd.Series([42]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert EarthMoversDistance(pd.Series([42]), pd.Series([1, 1, 1]), pd.Series([0, 0, 0])).distance == 0.75
    assert EarthMoversDistance(pd.Series([42]), pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])).distance == 0.375


def test_emd_categorical():
    assert EarthMoversDistanceCategorical(target, group1, group1).distance == 0
    assert EarthMoversDistanceCategorical(pd.Series([1, 0]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert EarthMoversDistanceCategorical(pd.Series([0, 1, 1]), pd.Series([1]), pd.Series([0, 1])).distance == 0.5
    assert EarthMoversDistanceCategorical(pd.Series([0, 0, 1, 1]), pd.Series([1, 1]), pd.Series([0, 0])).distance == 1


def test_ks_distance():
    assert KolmogorovSmirnovDistance(target, group1, group1).distance == 0
    assert KolmogorovSmirnovDistance(pd.Series([42]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert KolmogorovSmirnovDistance(pd.Series([42]), pd.Series([1, 1, 1]), pd.Series([0, 0, 0])).distance == 1
    assert KolmogorovSmirnovDistance(pd.Series([42]), pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])).distance == 0.5


def test_kl_divergence():
    assert KullbackLeiblerDivergence(target, group1, group1).distance == 0
    assert KullbackLeiblerDivergence(pd.Series([1, 0]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert KullbackLeiblerDivergence(pd.Series([0, 0, 1, 1]), pd.Series([1, 1]), pd.Series([0, 0])).distance == float(
        "inf"
    )


def test_js_divergence():
    assert JensenShannonDivergence(target, target, target).distance == 0
    assert JensenShannonDivergence(pd.Series([1, 0]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0

    js_group1_all = JensenShannonDivergence(target, group1, target).distance
    kl_group1_all = KullbackLeiblerDivergence(target, group1, target).distance
    assert js_group1_all * 2 == kl_group1_all


def test_norm():
    assert LNorm(target, group1, group1).distance == 0
    assert LNorm(pd.Series([1, 0]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert LNorm(pd.Series([1, 0]), pd.Series([1]), pd.Series([0]), ord=1).distance == 2
    assert LNorm(pd.Series(np.arange(10)), pd.Series(np.arange(5)), pd.Series(np.arange(5, 10)), ord=1).distance == 2


def test_hellinger():
    assert HellingerDistance(target, group1, group1).distance == 0
    assert HellingerDistance(pd.Series([1, 0]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert HellingerDistance(pd.Series([1, 0]), pd.Series([1]), pd.Series([0])).distance == 1
