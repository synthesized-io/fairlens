import pandas as pd

from fairlens.bias import metrics

df_compas = pd.read_csv("datasets/compas.csv")
group1_compas = df_compas[df_compas["Ethnicity"] == "Caucasian"]["RawScore"]
group2_compas = df_compas[(df_compas["Ethnicity"] == "African-American") | (df_compas["Ethnicity"] == "African-Am")]
target_compas = df_compas["RawScore"]


def test_class_imbalance():
    assert metrics.ClassImbalance(target_compas, group1_compas, group1_compas).distance == 0
    assert metrics.ClassImbalance(pd.Series([1, 2, 1]), pd.Series([1, 2]), pd.Series([1])).distance == 0.5
    assert metrics.ClassImbalance(pd.Series([1, 2, 3, 4]), pd.Series([1, 2]), pd.Series([3, 4])).distance == 0


def test_binomial_distance():
    assert metrics.BinomialDistance(target_compas, group1_compas, group1_compas).distance == 0
    assert metrics.BinomialDistance(pd.Series([42]), pd.Series([1, 1]), pd.Series([0, 0])).distance == 1
    assert metrics.BinomialDistance(pd.Series([42]), pd.Series([1, 0]), pd.Series([1, 0])).distance == 0
    assert metrics.BinomialDistance(pd.Series([42]), pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])).distance == 0.5


def test_emd():
    assert metrics.EarthMoversDistance(target_compas, group1_compas, group1_compas).distance == 0


def test_emd_categorical():
    assert metrics.EarthMoversDistanceCategorical(target_compas, group1_compas, group1_compas).distance == 0


def test_ks_distance():
    assert metrics.KolmogorovSmirnovDistance(target_compas, group1_compas, group1_compas).distance == 0


def test_kl_divergence():
    assert metrics.KullbackLeiblerDivergence(target_compas, group1_compas, group1_compas).distance == 0


def test_js_divergence():
    assert metrics.JensenShannonDivergence(target_compas, target_compas, target_compas).distance == 0


def test_norm():
    assert metrics.LNorm(target_compas, group1_compas, group1_compas).distance == 0


def test_hellinger():
    assert metrics.HellingerDistance(target_compas, group1_compas, group1_compas).distance == 0
