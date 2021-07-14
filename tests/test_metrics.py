import numpy as np
import pandas as pd

from fairlens.bias.metrics import BinomialDistance
from fairlens.bias.metrics import EarthMoversDistance as EMD
from fairlens.bias.metrics import EarthMoversDistanceCategorical as EMDC
from fairlens.bias.metrics import HellingerDistance
from fairlens.bias.metrics import JensenShannonDivergence as JS
from fairlens.bias.metrics import KolmogorovSmirnovDistance as KS
from fairlens.bias.metrics import KruskalWallis as KW
from fairlens.bias.metrics import KullbackLeiblerDivergence as KL
from fairlens.bias.metrics import MeanDistance as Mean
from fairlens.bias.metrics import Norm, stat_distance

df = pd.read_csv("datasets/compas.csv")
pred1 = df["Ethnicity"] == "Caucasian"
pred2 = df["Ethnicity"] == "African-American"
target_attr = "RawScore"
group1 = df[pred1][target_attr]
group2 = df[pred2][target_attr]
group3 = group2.sort_values()[len(group2) // 2 :]
group4 = df[~pred1][target_attr]
target = df[target_attr]
epsilon = 1e-6


def test_stat_distance():
    x = {"Ethnicity": ["Caucasian"]}
    y = {"Ethnicity": [e for e in list(df["Ethnicity"].unique()) if e != "Caucasian"]}
    xy = {"Ethnicity": list(df["Ethnicity"].unique())}

    res = EMD()(group1, target)
    assert stat_distance(df, target_attr, group1, target, mode="emd")[0] == res
    assert stat_distance(df, target_attr, x, xy, mode="emd")[0] == res

    res = stat_distance(df, target_attr, x, y, mode="emd")[0]
    assert stat_distance(df, target_attr, group1, group4, mode="emd")[0] == res


def test_stat_distance_auto():
    res = stat_distance(df, target_attr, group1, group2, mode="auto")[0]
    assert stat_distance(df, target_attr, group1, group2, mode="ks_distance")[0] == res


def test_auto_binning():
    res = stat_distance(df, target_attr, group1, group2, mode="emd_categorical")[0]
    assert stat_distance(df, target_attr, group1, group2, mode="emd")[0] == res


def test_mean_distance():
    assert Mean()(pd.Series(np.arange(100)), pd.Series(np.arange(10))) == 45

    assert Mean()(group1, group1) == 0
    assert Mean()(group1, group3) > Mean()(group1, group2)


def test_binomial_distance():
    assert BinomialDistance()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert BinomialDistance()(pd.Series([1, 1]), pd.Series([0, 0])) == 1
    assert BinomialDistance()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.5

    assert BinomialDistance()(pd.Series([True, False]), pd.Series([True, False])) == 0
    assert BinomialDistance()(pd.Series([False, False]), pd.Series([True, True])) == -1
    assert BinomialDistance()(pd.Series([True, False, True, True]), pd.Series([True, False, False, False])) == 0.5


def test_emd():
    assert EMD()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert EMD()(pd.Series([1, 1, 1]), pd.Series([0, 0, 0])) == 0.75
    assert EMD()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.375

    assert EMD()(group1, group1) == 0
    assert EMD()(group1, group3) > EMD()(group1, group2)


def test_ks_distance():
    assert KS()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert KS()(pd.Series([1, 1, 1]), pd.Series([0, 0, 0])) == 1
    assert KS()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.5

    assert KS()(group1, group1) == 0
    assert KS()(group1, group3) > KS()(group1, group2)


def test_kruskal_wallis():
    assert KW()(pd.Series([1, 0]), pd.Series([1, 0])) == 0

    assert abs(KW()(group1, group1)) < 1e-6


def test_emd_categorical():
    assert EMDC()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert EMDC()(pd.Series([1]), pd.Series([0, 1])) == 0.5
    assert EMDC()(pd.Series([1, 1]), pd.Series([0, 0])) == 1

    assert EMDC()(pd.Series(["b"]), pd.Series(["a", "b"])) == 0.5
    assert EMDC()(pd.Series(["b", "b"]), pd.Series(["a", "a"])) == 1

    assert EMDC()(group1, group1) == 0
    assert EMDC()(group1, group3) > EMDC()(group1, group2)

    assert EMDC(bin_edges=[0, 1])(group1, group2) == 0
    assert EMDC()(group1, group2) == EMD()(group1, group2)


def test_kl_divergence():
    assert KL()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert KL()(pd.Series([1, 1]), pd.Series([0, 0])) == float("inf")

    assert KL()(group1, group1) == 0


def test_js_divergence():
    assert JS()(pd.Series([1, 0]), pd.Series([1, 0])) == 0

    assert JS()(group1, group1) == 0
    assert JS()(group1, group3) > JS()(group1, group2)


def test_norm():
    assert Norm()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert Norm(ord=1)(pd.Series([1]), pd.Series([0])) == 2
    assert Norm(ord=1)(pd.Series(np.arange(5)), pd.Series(np.arange(5, 10))) == 2

    assert Norm()(group1, group1) == 0
    assert Norm()(group1, group3) > Norm()(group1, group2)


def test_hellinger():
    assert HellingerDistance()(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert HellingerDistance()(pd.Series([1]), pd.Series([0])) == 1

    assert HellingerDistance()(group1, group1) == 0
    assert HellingerDistance()(group1, group3) > HellingerDistance()(group1, group2)
