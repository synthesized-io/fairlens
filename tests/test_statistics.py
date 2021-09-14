import datetime

import numpy as np
import pandas as pd
import pytest
from scipy.stats import describe, entropy

import fairlens.metrics.statistics as fls


def test_distribution_mean_continuous():
    sr = pd.Series(np.random.randn(50))
    assert fls.compute_distribution_mean(sr) == describe(sr).mean


def test_distribution_mean_categorical_mode_square():
    sr = pd.Series(
        [
            "African-American",
            "Caucasian",
            "African-American",
            "Caucasian",
            "Hispanic",
            "African-American",
            "African-American",
            "Other",
            "Hispanic",
            "Caucasian",
        ]
    )
    assert fls.compute_distribution_mean(sr, categorical_mode="square") == "African-American"


def test_distribution_mean_categorical_mode_entropy():
    sr = pd.Series(
        [
            "African-American",
            "Caucasian",
            "African-American",
            "Caucasian",
            "Other",
            "Caucasian",
            "African-American",
            "Other",
            "Hispanic",
            "Caucasian",
        ]
    )
    assert fls.compute_distribution_mean(sr, categorical_mode="entropy") == "Caucasian"


def test_distribution_mean_categorical_mode_means():
    sr = pd.Series(
        [
            "African-American",
            "Caucasian",
            "African-American",
            "Caucasian",
            "Hispanic",
            "African-American",
            "African-American",
            "Other",
            "Hispanic",
            "Caucasian",
            "Other",
            "Caucasian",
            "African-American",
            "Hispanic",
        ]
    )
    res = fls.compute_distribution_mean(sr, categorical_mode="multinomial")
    for val in sr.unique().tolist():
        assert res[val] == sr.str.count(val).sum() / float(sr.size)


def test_distribution_mean_datetime():
    sr = pd.Series(["24/3/2000", "27/4/1999", "4/2/1998"] * 5)
    assert fls.compute_distribution_mean(sr) == datetime.datetime(1999, 4, 7, 16, 0, 0)


def test_distribution_variance_categorical_square():
    sr = pd.Series(
        [
            "African-American",
            "Caucasian",
            "African-American",
            "Caucasian",
            "Hispanic",
            "African-American",
            "African-American",
            "Other",
            "Hispanic",
            "Caucasian",
        ]
    )
    assert fls.compute_distribution_variance(sr, categorical_mode="square") == pytest.approx(0.3)


def test_distribution_variance_categorical_entropy():
    sr = pd.Series(
        [
            "African-American",
            "Caucasian",
            "African-American",
            "Caucasian",
            "Other",
            "Caucasian",
            "African-American",
            "Other",
            "Hispanic",
            "Caucasian",
        ]
    )
    assert fls.compute_distribution_variance(sr, categorical_mode="entropy") == entropy(sr.value_counts())


def test_distribution_variance_categorical_multinomial():
    sr = pd.Series(
        [
            "African-American",
            "Caucasian",
            "African-American",
            "Caucasian",
            "Hispanic",
            "African-American",
            "African-American",
            "Other",
            "Hispanic",
            "Caucasian",
            "Other",
            "Caucasian",
            "African-American",
            "Hispanic",
        ]
    )
    res = fls.compute_distribution_variance(sr, categorical_mode="multinomial")
    for val in sr.unique().tolist():
        prob = sr.str.count(val).sum() / float(sr.size)
        assert res[val] == prob * (1 - prob)


def test_distribution_variance_continuous():
    sr = pd.Series(np.random.randn(50))
    assert fls.compute_distribution_variance(sr) == describe(sr).variance


def test_sensitive_analysis_target_numeric():
    data = [
        [10, "Caucasian", "24/4/1999"],
        [25, "African-American", "19/7/1997"],
        [15, "Hispanic", "31/12/2001"],
        [34, "Other", "20/2/1998"],
        [35, "Caucasian", "2/3/2002"],
        [56, "Hispanic", "6/6/1997"],
        [80, "African-American", "4/5/2000"],
        [100, "African-American", "3/1/1996"],
    ]
    res = fls.sensitive_group_analysis(
        pd.DataFrame(data=data, columns=["score", "race", "date"]),
        target_attr="score",
        groups=[{"race": ["African-American"]}],
    )
    assert res["Means"][0] == 25
    assert res["Variances"][0] == 0.5
