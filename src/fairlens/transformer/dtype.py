from typing import Optional

import pandas as pd

from .base import Transformer


class DTypeTransformer(Transformer):
    """
    Infer hidden dtype of data and convert data to it.

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str, out_dtype: Optional[str] = None):
        super().__init__(name=name)
        self.in_dtype: Optional[str] = None
        self.out_dtype: Optional[str] = out_dtype

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes}, out_dtype={self.out_dtype})"

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        self.in_dtype = str(df[self.name].dtype)
        if self.out_dtype is not None:
            return super().fit(df)

        column = df[self.name].copy()

        # Try to convert it to numeric
        if column.dtype.kind not in ("i", "u", "f"):
            n_nans = column.isna().sum()
            col_num = pd.to_numeric(column, errors="coerce")
            if col_num.isna().sum() == n_nans:
                column = col_num

        # Try to convert it to date
        if column.dtype.kind == "O":
            n_nans = column.isna().sum()
            try:
                col_date = pd.to_datetime(column, errors="coerce")
            except TypeError:  # Argument 'date_string' has incorrect type (expected str, got numpy.str_)
                col_date = pd.to_datetime(column.astype(str), errors="coerce")

            if col_date.isna().sum() == n_nans:
                column = col_date

        self.out_dtype = str(column.dtype)
        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()

        if self.out_dtype == self.in_dtype:
            return df
        elif self.out_dtype in ("i", "u", "f", "f8", "i8", "u8"):
            df[self.name] = pd.to_numeric(df[self.name], errors="coerce")
        else:
            df[self.name] = df[self.name].astype(self.out_dtype, errors="ignore")
        return df
