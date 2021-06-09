import pandas as pd
from fairlens.bias import p_value as pv
from fairlens.bias.metrics import stat_distance


epsilon = 1e-5
df = pd.read_csv("datasets/compas.csv")
group1 = df[df["Ethnicity"] == "Caucasian"]["RawScore"]
group2 = df[df["Ethnicity"] == "African-American"]["RawScore"]
target = df["RawScore"]


def test_binomial():
    assert abs(pv.binominal_proportion_p_value(0.2, 0.1, 10) - (1 - (0.9 ** 10 + 0.9 ** 9))) < epsilon

def test_bootstrapping():
    pass

def test_permutation():
    pass

def test_ks_distance():
    pass