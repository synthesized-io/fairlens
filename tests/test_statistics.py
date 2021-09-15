import datetime

import numpy as np
import pandas as pd
import pytest
from scipy.stats import describe, entropy

import fairlens.metrics.statistics as fls


def test_distribution_mean_continuous():
    sr = pd.Series(np.random.randn(50))
    assert fls.compute_distribution_mean(sr, x_type="continuous") == describe(sr).mean


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
    assert fls.compute_distribution_mean(sr, x_type="categorical", categorical_mode="square") == "African-American"


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
    assert fls.compute_distribution_mean(sr, x_type="categorical", categorical_mode="entropy") == "Caucasian"


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
    res = fls.compute_distribution_mean(sr, x_type="categorical", categorical_mode="multinomial")
    for val in sr.unique().tolist():
        assert res[val] == sr.str.count(val).sum() / float(sr.size)


def test_distribution_mean_datetime():
    sr = pd.Series(["24/3/2000", "27/4/1999", "4/2/1998"] * 5)
    assert fls.compute_distribution_mean(sr, x_type="datetime") == datetime.datetime(1999, 4, 7, 16, 0, 0)


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
    assert fls.compute_distribution_variance(sr, x_type="categorical", categorical_mode="square") == pytest.approx(0.3)


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
    assert fls.compute_distribution_variance(sr, x_type="categorical", categorical_mode="entropy") == entropy(
        sr.value_counts()
    )


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
    res = fls.compute_distribution_variance(sr, x_type="categorical", categorical_mode="multinomial")
    for val in sr.unique().tolist():
        prob = sr.str.count(val).sum() / float(sr.size)
        assert res[val] == prob * (1 - prob)


def test_distribution_variance_continuous():
    sr = pd.Series(np.random.randn(50))
    assert fls.compute_distribution_variance(sr, x_type="continuous") == describe(sr).variance


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
        [134, "Caucasian", "24/4/1999"],
        [21, "African-American", "19/7/1997"],
        [14, "Hispanic", "31/12/2001"],
        [98, "Other", "20/2/1998"],
        [76, "Caucasian", "2/3/2002"],
        [51, "Hispanic", "6/6/1997"],
        [82, "African-American", "4/5/2000"],
        [145, "African-American", "3/1/1996"],
    ]
    res = fls.sensitive_group_analysis(
        pd.DataFrame(data=data, columns=["score", "race", "date"]),
        target_attr="score",
        groups=[{"race": ["African-American"]}],
        categorical_mode="square",
    )
    assert res["Means"][0] == 75.5
    assert res["Variances"][0] == 2202.7


def test_sensitive_analysis_target_datetime():
    data = [
        [10, "Caucasian", "24/4/1999"],
        [25, "African-American", "19/7/1997"],
        [15, "Hispanic", "31/12/2001"],
        [34, "Other", "20/2/1998"],
        [35, "Caucasian", "2/3/2002"],
        [56, "Hispanic", "6/6/1997"],
        [80, "African-American", "4/5/2000"],
        [100, "African-American", "3/1/1996"],
        [134, "Caucasian", "24/4/1999"],
        [21, "African-American", "19/7/1997"],
        [14, "Hispanic", "31/12/2001"],
        [98, "Other", "20/2/1998"],
        [76, "Caucasian", "2/3/2002"],
        [51, "Hispanic", "6/6/1997"],
        [82, "African-American", "4/5/2000"],
        [145, "African-American", "3/1/1996"],
    ]
    res = fls.sensitive_group_analysis(
        pd.DataFrame(data=data, columns=["score", "race", "date"]),
        target_attr="date",
        groups=[{"race": ["African-American"]}],
        categorical_mode="square",
    )
    assert res["Means"][0] == datetime.datetime(1997, 12, 28)


def test_sensitive_analysis_target_cat_square():
    data = [
        [10, "Caucasian", "24/4/1999", "Rejected"],
        [25, "African-American", "19/7/1997", "Accepted"],
        [15, "Hispanic", "31/12/2001", "Accepted"],
        [34, "Other", "20/2/1998", "Rejected"],
        [35, "Caucasian", "2/3/2002", "Accepted"],
        [56, "Hispanic", "6/6/1997", "Accepted"],
        [80, "African-American", "4/5/2000", "Accepted"],
        [100, "African-American", "3/1/1996", "Accepted"],
        [134, "Caucasian", "24/4/1999", "Rejected"],
        [21, "African-American", "19/7/1997", "Rejected"],
        [14, "Hispanic", "31/12/2001", "Rejected"],
        [98, "Other", "20/2/1998", "Rejected"],
        [76, "Caucasian", "2/3/2002", "Accepted"],
        [51, "Hispanic", "6/6/1997", "Accepted"],
        [82, "African-American", "4/5/2000", "Rejected"],
        [145, "African-American", "3/1/1996", "Accepted"],
    ]
    res = fls.sensitive_group_analysis(
        pd.DataFrame(data=data, columns=["score", "race", "date", "status"]),
        target_attr="status",
        groups=[{"race": ["Caucasian", "Hispanic"]}],
        categorical_mode="square",
    )
    assert res["Means"][0] == "Accepted"
    assert res["Variances"][0] == 0.53125


def test_sensitive_analysis_target_cat_entropy():
    data = [
        [10, "Caucasian", "24/4/1999", "Rejected"],
        [25, "African-American", "19/7/1997", "Rejected"],
        [15, "Hispanic", "31/12/2001", "Accepted"],
        [34, "Other", "20/2/1998", "Rejected"],
        [35, "Caucasian", "2/3/2002", "Accepted"],
        [56, "Hispanic", "6/6/1997", "Accepted"],
        [80, "African-American", "4/5/2000", "Accepted"],
        [100, "African-American", "3/1/1996", "Rejected"],
        [134, "Caucasian", "24/4/1999", "Rejected"],
        [21, "African-American", "19/7/1997", "Rejected"],
        [14, "Hispanic", "31/12/2001", "Rejected"],
        [98, "Other", "20/2/1998", "Rejected"],
        [76, "Caucasian", "2/3/2002", "Accepted"],
        [51, "Hispanic", "6/6/1997", "Accepted"],
        [82, "African-American", "4/5/2000", "Rejected"],
        [145, "African-American", "3/1/1996", "Accepted"],
    ]
    res = fls.sensitive_group_analysis(
        pd.DataFrame(data=data, columns=["score", "race", "date", "status"]),
        target_attr="status",
        groups=[{"race": ["African-American", "Other"]}],
        categorical_mode="entropy",
    )
    assert res["Means"][0] == "Rejected"
    assert res["Variances"][0] == pytest.approx(0.5623, rel=1e-3)
