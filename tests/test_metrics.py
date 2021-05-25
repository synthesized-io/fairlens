import pandas as pd

from fairlens.bias.metrics import class_imbalance, emd, js_divergence, kl_divergence, ks_distance, lp_norm

df_compas = pd.read_csv("datasets/compas.csv")


def test_class_imbalance():
    assert class_imbalance(df_compas, "RawScore", {"Ethnicity": ["Caucasian"]}, {"Ethnicity": ["Caucasian"]}) == 0


def test_emd():
    assert emd(df_compas, "RawScore", {"Ethnicity": ["Caucasian"]}, {"Ethnicity": ["Caucasian"]}) == 0


def test_ks_distance():
    assert ks_distance(df_compas, "RawScore", {"Ethnicity": ["Caucasian"]}, {"Ethnicity": ["Caucasian"]}) == (0, 1)


def test_kl_divergence():
    assert kl_divergence(df_compas, "RawScore", {"Ethnicity": ["Caucasian"]}, {"Ethnicity": ["Caucasian"]}) == 0


def test_js_divergence():
    assert js_divergence(df_compas, "RawScore", {"Ethnicity": ["Caucasian"]}, {"Ethnicity": ["Caucasian"]}) > 0


def test_lp_norm():
    assert lp_norm(df_compas, "RawScore", {"Ethnicity": ["Caucasian"]}, {"Ethnicity": ["Caucasian"]}) == 0
