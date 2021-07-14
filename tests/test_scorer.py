import pandas as pd

from fairlens.scorer.fairness_scorer import FairnessScorer

dfc = pd.read_csv("datasets/compas.csv")


def test_fairness_scorer_runs():
    fscorer = FairnessScorer(dfc, "RawScore", ["Ethnicity", "Sex"])
    assert fscorer.sensitive_attrs == ["Ethnicity", "Sex"]
    assert fscorer.target_attr == "RawScore"

    score = fscorer.distribution_score(alpha=0.95)[0]
    assert score > 0


def test_sensitive_attr_detection():
    fscorer = FairnessScorer(dfc, "RawScore")
    assert sorted(fscorer.sensitive_attrs) == ["DateOfBirth", "Ethnicity", "Language", "MaritalStatus", "Sex"]

    fscorer = FairnessScorer(dfc, "RawScore", ["Arbitrary"], detect_sensitive=True)
    assert sorted(fscorer.sensitive_attrs) == [
        "Arbitrary",
        "DateOfBirth",
        "Ethnicity",
        "Language",
        "MaritalStatus",
        "Sex",
    ]


def test_distribution_score():
    fscorer = FairnessScorer(dfc, "RawScore", ["Ethnicity", "Sex"])
    score, df_dist = fscorer.distribution_score()

    assert score * df_dist["Counts"].sum() == (df_dist["Distance"] * df_dist["Counts"]).sum()
