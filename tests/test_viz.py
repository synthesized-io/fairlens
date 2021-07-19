import pandas as pd

from fairlens.bias.viz import attr_distr_plot, distr_pair_plot, distr_plot, mult_distr_plot

dfc = pd.read_csv("datasets/compas.csv")


def test_distr_pair_plot():
    distr_pair_plot(dfc, "RawScore", {"Sex": ["Male"]}, {"Sex": ["Female"]})


def test_distr_plot():
    distr_plot(dfc, "RawScore", [{"Sex": ["Male"]}, {"Sex": ["Female"]}, {"Ethnicity": ["Asian"]}])
    distr_plot(dfc, "RawScore", [{"Sex": ["Male"]}, dfc["Sex"] == "Female"])


def test_attr_distr_plot():
    attr_distr_plot(dfc, "RawScore", "Sex")
    attr_distr_plot(dfc, "RawScore", "Ethnicity", separate=True)


def test_mult_distr_plot():
    mult_distr_plot(dfc, "RawScore", ["Sex", "DateOfBirth", "Ethnicity", "MaritalStatus", "Language"])