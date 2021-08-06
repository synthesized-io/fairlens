import pandas as pd

from fairlens.scorer import FairnessScorer, calculate_score

dfa = pd.read_csv("datasets/adult.csv")
dfc = pd.read_csv("datasets/compas.csv")
dfg = pd.read_csv("datasets/german_credit_data.csv")
dft = pd.read_csv("datasets/titanic.csv")


def test_fairness_scorer_runs_compas():
    fscorer = FairnessScorer(dfc, "RawScore", ["DateOfBirth", "Ethnicity", "Sex"])
    assert fscorer.sensitive_attrs == ["DateOfBirth", "Ethnicity", "Sex"]

    _ = fscorer.plot_distributions()
    df_dist = fscorer.distribution_score()
    score = calculate_score(df_dist)
    assert score > 0


def test_fairness_scorer_runs_german():
    fscorer = FairnessScorer(dfg, "Credit amount")
    assert fscorer.sensitive_attrs == ["Age", "Sex"]

    _ = fscorer.plot_distributions()
    df_dist = fscorer.distribution_score()
    score = calculate_score(df_dist)
    assert score > 0


def test_fairness_scorer_runs_adult():
    fscorer = FairnessScorer(dfa, "class")
    assert fscorer.sensitive_attrs == ["age", "marital-status", "race", "relationship", "sex"]

    fscorer = FairnessScorer(dfa, "class", ["age", "race", "sex"])

    _ = fscorer.plot_distributions()
    df_dist = fscorer.distribution_score()
    score = calculate_score(df_dist)
    assert score > 0


def test_fairness_scorer_runs_titanic():
    fscorer = FairnessScorer(dft, "Survived")
    assert fscorer.sensitive_attrs == ["Age", "Sex"]

    _ = fscorer.plot_distributions()
    df_dist = fscorer.distribution_score()
    score = calculate_score(df_dist)
    assert score > 0


def test_sensitive_attr_detection():
    fscorer = FairnessScorer(dfc, "RawScore")
    assert fscorer.sensitive_attrs == ["DateOfBirth", "Ethnicity", "Language", "MaritalStatus", "Sex"]

    fscorer = FairnessScorer(dfc, "RawScore", ["RawScore"], detect_sensitive=True)
    assert fscorer.sensitive_attrs == ["DateOfBirth", "Ethnicity", "Language", "MaritalStatus", "RawScore", "Sex"]


def test_distribution_score():
    fscorer = FairnessScorer(dfc, "RawScore", ["Ethnicity", "Sex"])
    df_dist = fscorer.distribution_score()
    score = calculate_score(df_dist)

    assert score * df_dist["Counts"].sum() == (df_dist["Distance"] * df_dist["Counts"]).sum()
