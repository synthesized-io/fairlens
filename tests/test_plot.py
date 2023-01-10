import pandas as pd
import seaborn as sns

from fairlens.plot.distr import attr_distr_plot, distr_plot, mult_distr_plot

dfa = pd.read_csv("https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/templates/adult.csv")
dfc = pd.read_csv("https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/biased/compas.csv")
dfg = pd.read_csv(
    "https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/templates/german_credit_data.csv"
)
dft = pd.read_csv("https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/templates/titanic.csv")


def test_distr_plot():
    distr_plot(dfc, "RawScore", [{"Sex": ["Male"]}, {"Sex": ["Female"]}, {"Ethnicity": ["Asian"]}])

    groups = [{"Sex": ["Male"]}, dfc["Sex"] == "Female"]
    distr_plot(dfc, "RawScore", groups, cmap=sns.color_palette())
    distr_plot(dfc, "RawScore", groups, show_curve=None)
    distr_plot(dfc, "RawScore", groups, show_hist=True, show_curve=False)
    distr_plot(dfc, "RawScore", groups, show_hist=False, show_curve=True)
    distr_plot(dfc, "RawScore", groups, normalize=True)
    distr_plot(dfc, "RawScore", groups, normalize=True, distr_type="continuous")
    distr_plot(dfc, "DateOfBirth", groups, normalize=True, distr_type="datetime")


def test_attr_distr_plot():
    attr_distr_plot(dfc, "RawScore", "Sex")
    attr_distr_plot(dfc, "RawScore", "Sex", distr_type="continuous", attr_distr_type="binary")
    attr_distr_plot(dfc, "RawScore", "Ethnicity", separate=True)


def test_mult_distr_plot_compas():
    mult_distr_plot(dfc, "RawScore", ["Sex", "DateOfBirth", "Ethnicity", "MaritalStatus", "Language"])


def test_mult_distr_plot_adult():
    mult_distr_plot(dfa, "income", ["age", "marital-status", "relationship", "race", "gender"])


def test_mult_distr_plot_german():
    mult_distr_plot(dfg, "Credit amount", ["Sex", "Age"])


def test_mult_distr_plot_titanic():
    mult_distr_plot(dft, "Survived", ["Sex", "Age"])
