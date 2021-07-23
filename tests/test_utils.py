import numpy as np
import pandas as pd

from fairlens.bias import utils

dfc = pd.read_csv("datasets/compas.csv")


def test_zipped_hist():
    arr1 = np.arange(10)
    arr2 = np.arange(5, 10)
    hist1, hist2 = utils.zipped_hist((pd.Series(arr1), pd.Series(arr2)))
    assert (hist1 == np.bincount(arr1) / len(arr1)).all()
    assert (hist2 == np.bincount(arr2) / len(arr2)).all()

    arr = np.concatenate([np.arange(10)] * 10)
    assert (utils.zipped_hist((pd.Series(arr),))[0] == np.bincount(arr) / len(arr)).all()

    arr = np.random.rand(1000)
    hist, bin_edges = utils.zipped_hist((pd.Series(arr),), ret_bins=True)
    _hist, _bin_edges = np.histogram(arr, bins="auto")
    assert (hist == _hist / _hist.sum()).all() and (bin_edges == _bin_edges).all()


def test_bin():
    columns = ["A", "B", "C"]

    df = pd.DataFrame(np.array([np.arange(101) * (i + 1) for i in range(3)]).T, index=range(101), columns=columns)
    assert df.loc[:, "A"].nunique() > 4

    a_binned = utils.bin(df["A"], 4, duplicates="drop", remove_outliers=0.1)
    assert a_binned.nunique() == 4


def test_quantize_dates():
    col = pd.Series(pd.date_range(start="1/1/2018", periods=5))
    assert utils.quantize_date(col).equals(pd.Series(["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]))

    col = pd.Series(pd.to_datetime(["1/1/1999", "1/1/2020"]))
    assert utils.quantize_date(col).equals(pd.Series(["1990-2000", "2020-2030"]))

    col = pd.Series(pd.date_range(start="1/1/2018", periods=70))
    assert (utils.quantize_date(col).unique() == ["Jan", "Feb", "Mar"]).all()

    col = pd.Series(pd.date_range(start="1/1/2018", periods=500))
    assert (utils.quantize_date(col).unique() == [2018, 2019]).all()

    col = pd.Series(pd.date_range(start="1/1/2018", periods=2000))
    assert (utils.quantize_date(col).unique() == [2018, 2019, 2020, 2021, 2022, 2023]).all()

    col = pd.Series(pd.date_range(start="1/1/2018", periods=7000))
    assert (utils.quantize_date(col).unique() == ["2010-2020", "2020-2030", "2030-2040"]).all()


def test_infer_dtype():
    cols = ["A", "B", "C"]
    df = pd.DataFrame(np.array([np.arange(11) * (i + 1) for i in range(len(cols))]).T, index=range(11), columns=cols)
    assert str(utils.infer_dtype(df["A"]).dtype) == "int64"

    df = pd.DataFrame(
        np.array([np.linspace(0, 10, 21) * (i + 1) for i in range(len(cols))]).T, index=range(21), columns=cols
    )
    assert str(utils.infer_dtype(df["A"]).dtype) == "float64"


def test_infer_distr_type():
    assert utils.infer_distr_type(pd.Series(np.linspace(-20, 20, 200))).is_continuous()
    assert utils.infer_distr_type(pd.Series(np.linspace(-20, 20, 9))).is_categorical()
    assert utils.infer_distr_type(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])).is_continuous()
    assert utils.infer_distr_type(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).is_categorical()
    assert utils.infer_distr_type(pd.Series([1, 0] * 10)).is_binary()
    assert utils.infer_distr_type(pd.Series([1, 0, 1, 1])).is_binary()
    assert utils.infer_distr_type(pd.Series([True, False, True, True])).is_binary()
    assert utils.infer_distr_type(pd.Series([1, 1, 1])).is_categorical()
    assert utils.infer_distr_type(pd.Series([0])).is_categorical()
    assert utils.infer_distr_type(pd.Series([2, 2])).is_categorical()
    assert utils.infer_distr_type(pd.Series(["hi", "hello"])).is_categorical()
    assert utils.infer_distr_type(pd.Series(["n" + str(i) for i in np.arange(20)])).is_categorical()


def test_get_predicates_mult():
    cols = ["A", "B", "C"]
    df = pd.DataFrame(np.array([np.arange(11) * (i + 1) for i in range(len(cols))]).T, index=range(11), columns=cols)

    preds = utils.get_predicates_mult(df, [{"A": [1], "B": [2]}, {"C": np.arange(10) * 3 + 1}])

    assert df[preds[0]]["C"].nunique() == 1
    assert df[preds[1]]["C"].nunique() == 0

    preds = utils.get_predicates_mult(
        dfc, [{"Ethnicity": ["African-American"], "Sex": ["Male"]}, {"Ethnicity": ["Caucasian"]}]
    )
    pred1, pred2 = preds[0], preds[1]

    assert dfc[pred1].equals(dfc[(dfc["Ethnicity"] == "African-American") & (dfc["Sex"] == "Male")])
    assert dfc[pred2].equals(dfc[dfc["Ethnicity"] == "Caucasian"])

    preds = utils.get_predicates_mult(dfc, [{"Sex": ["Male"]}, dfc["Ethnicity"] == "African-American"])
    pred1, pred2 = preds[0], preds[1]

    assert dfc[pred1].equals(dfc[dfc["Sex"] == "Male"])
    assert dfc[pred2].equals(dfc[dfc["Ethnicity"] == "African-American"])
