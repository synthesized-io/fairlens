import numpy as np
import pandas as pd

from fairlens.bias import utils


def test_bin():
    columns = ["A", "B", "C"]

    df = pd.DataFrame(np.array([np.arange(101) * (i + 1) for i in range(3)]).T, index=range(101), columns=columns)
    assert df.loc[:, "A"].nunique() > 4

    df = utils.bin(df, "A", 4, duplicates="drop", remove_outliers=0.1)
    assert df.loc[:, "A"].nunique() == 4


def test_infer_distr_type():
    assert utils.infer_distr_type(pd.Series(np.linspace(-20, 20, 200))).is_continuous()
    assert utils.infer_distr_type(pd.Series(np.linspace(-20, 20, 9))).is_categorical()
    assert utils.infer_distr_type(pd.Series([1, 2] * 10)).is_binary()
