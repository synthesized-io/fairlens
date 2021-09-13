import pandas as pd
import seaborn as sns

from fairlens.plot.correlation import heatmap
from fairlens.plot.distr import attr_distr_plot, distr_plot, mult_distr_plot

dfa = pd.read_csv("datasets/adult.csv")
dfc = pd.read_csv("datasets/compas.csv")
dfg = pd.read_csv("datasets/german_credit_data.csv")
dft = pd.read_csv("datasets/titanic.csv")


def test_distr_plot():
    distr_plot(dfc, "RawScore", [{"Sex": ["Male"]}, {"Sex": ["Female"]}, {"Ethnicity": ["Asian"]}])
    distr_plot(dfc, "RawScore", [{"Sex": ["Male"]}, dfc["Sex"] == "Female"], cmap=sns.color_palette())


def test_attr_distr_plot():
    attr_distr_plot(dfc, "RawScore", "Sex")
    attr_distr_plot(dfc, "RawScore", "Ethnicity", separate=True)


def test_mult_distr_plot_compas():
    mult_distr_plot(dfc, "RawScore", ["Sex", "DateOfBirth", "Ethnicity", "MaritalStatus", "Language"])


def test_mult_distr_plot_adult():
    mult_distr_plot(dfa, "class", ["age", "marital-status", "relationship", "race", "sex"])


def test_mult_distr_plot_german():
    mult_distr_plot(dfg, "Credit amount", ["Sex", "Age"])


def test_mult_distr_plot_titanic():
    mult_distr_plot(dft, "Survived", ["Sex", "Age"])


def test_heatmap_adult():
    heatmap(dfa)


def test_heatmap_german():
    heatmap(dfg)


def test_heatmap_titanic():
    heatmap(dft)
