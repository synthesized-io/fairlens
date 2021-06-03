import numpy as np
import pandas as pd

from fairlens.bias import utils

dfc = pd.read_csv("datasets/compas.csv")


def test_bin():
    columns = ["A", "B", "C"]

    df = pd.DataFrame(np.array([np.arange(101) * (i + 1) for i in range(3)]).T, index=range(101), columns=columns)
    assert df.loc[:, "A"].nunique() > 4

    A_binned = utils.bin(df["A"], 4, duplicates="drop", remove_outliers=0.1)
    assert A_binned.nunique() == 4


def test_infer_dtype():
    cols = ["A", "B", "C"]
    df = pd.DataFrame(np.array([np.arange(11) * (i + 1) for i in range(len(cols))]).T, index=range(11), columns=cols)
    assert str(utils.infer_dtype(df, "A")["A"].dtype) == "int64"

    df = pd.DataFrame(
        np.array([np.linspace(0, 10, 21) * (i + 1) for i in range(len(cols))]).T, index=range(21), columns=cols
    )
    assert str(utils.infer_dtype(df, "A")["A"].dtype) == "float64"


def test_infer_distr_type():
    assert utils.infer_distr_type(pd.Series(np.linspace(-20, 20, 200))).is_continuous()
    assert utils.infer_distr_type(pd.Series(np.linspace(-20, 20, 9))).is_categorical()
    assert utils.infer_distr_type(pd.Series([1, 2] * 10)).is_binary()


def test_get_predicates():
    cols = ["A", "B", "C"]
    df = pd.DataFrame(np.array([np.arange(11) * (i + 1) for i in range(len(cols))]).T, index=range(11), columns=cols)

    pred1, pred2 = utils.get_predicates(df, {"A": [1], "B": [2]}, {"C": np.arange(10) * 3 + 1})

    assert df[pred1]["C"].nunique() == 1
    assert df[pred2]["C"].nunique() == 0


def test_get_predicates_mult():
    cols = ["A", "B", "C"]
    df = pd.DataFrame(np.array([np.arange(11) * (i + 1) for i in range(len(cols))]).T, index=range(11), columns=cols)

    preds = utils.get_predicates_mult(df, [{"A": [1], "B": [2]}, {"C": np.arange(10) * 3 + 1}])

    assert df[preds[0]]["C"].nunique() == 1
    assert df[preds[1]]["C"].nunique() == 0

    preds = utils.get_predicates_mult(
        dfc, [{"Ethnicity": ["African-American", "African-Am"], "Sex": ["Male"]}, {"Ethnicity": ["Caucasian"]}]
    )
    pred1, pred2 = preds[0], preds[1]

    assert dfc[pred1].equals(
        dfc[((dfc["Ethnicity"] == "African-American") | (dfc["Ethnicity"] == "African-Am")) & (dfc["Sex"] == "Male")]
    )
    assert dfc[pred2].equals(dfc[dfc["Ethnicity"] == "Caucasian"])


def test_align_probabilities():
    space = [1, 2, 3, 4]
    counts = {2: 3, 3: 2}

    p = utils.align_probabilities(counts, space)

    assert (p == [0, 0.6, 0.4, 0]).all()
    assert not (p == [0, 0.4, 0.6, 0]).all()
