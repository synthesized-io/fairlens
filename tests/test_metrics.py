import numpy as np
import pandas as pd

from fairlens.bias.metrics import (  # isort:skip
    BinomialDistance,
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

    res = EarthMoversDistance()(group1, target)
    assert stat_distance(df, target_attr, group1, target, mode="emd") == res
    assert stat_distance(df, target_attr, x, xy, mode="emd") == res

    res = stat_distance(df, target_attr, x, mode="emd")
    assert stat_distance(df, target_attr, df[pred1][target_attr], df[~pred1][target_attr], mode="emd") == res
    assert stat_distance(df, target_attr, x, y, mode="emd") == res


def test_stat_distance_auto():
    res = stat_distance(df, target_attr, group1, group2, mode="auto")
    assert stat_distance(df, target_attr, group1, group2, mode="ks_distance") == res


def test_auto_binning():
    res = stat_distance(df, target_attr, group1, group2, mode="emd_categorical")
    assert stat_distance(df, target_attr, group1, group2, mode="emd") == res


def test_binomial_distance():
    assert BinomialDistance()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert BinomialDistance()(pd.Series([1, 1]), pd.Series([0, 0])) == 1
    assert BinomialDistance()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.5


def test_emd():
    assert EarthMoversDistance()(group1, group1) == 0
    assert EarthMoversDistance()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert EarthMoversDistance()(pd.Series([1, 1, 1]), pd.Series([0, 0, 0])) == 0.75
    assert EarthMoversDistance()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.375


def test_emd_categorical():
    assert EarthMoversDistanceCategorical()(group1, group1) == 0
    assert EarthMoversDistanceCategorical()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert EarthMoversDistanceCategorical()(pd.Series([1]), pd.Series([0, 1])) == 0.5
    assert EarthMoversDistanceCategorical()(pd.Series([1, 1]), pd.Series([0, 0])) == 1


def test_ks_distance():
    assert KolmogorovSmirnovDistance()(group1, group1) == 0
    assert KolmogorovSmirnovDistance()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert KolmogorovSmirnovDistance()(pd.Series([1, 1, 1]), pd.Series([0, 0, 0])) == 1
    assert KolmogorovSmirnovDistance()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.5


def test_kl_divergence():
    assert KullbackLeiblerDivergence()(group1, group1) == 0
    assert KullbackLeiblerDivergence()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert KullbackLeiblerDivergence()(pd.Series([1, 1]), pd.Series([0, 0])) == float("inf")


def test_js_divergence():
    assert JensenShannonDivergence()(group1, group1) == 0
    assert JensenShannonDivergence()(pd.Series([1, 0]), pd.Series([1, 0])) == 0


def test_norm():
    assert LNorm()(group1, group1) == 0
    assert LNorm()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert LNorm(ord=1)(pd.Series([1]), pd.Series([0])) == 2
    assert LNorm(ord=1)(pd.Series(np.arange(5)), pd.Series(np.arange(5, 10))) == 2


def test_hellinger():
    assert HellingerDistance()(group1, group1) == 0
    assert HellingerDistance()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert HellingerDistance()(pd.Series([1]), pd.Series([0])) == 1
