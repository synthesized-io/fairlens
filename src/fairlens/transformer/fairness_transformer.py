import logging
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..transformer import BinningTransformer, DTypeTransformer, Transformer

logger = logging.getLogger(__name__)


class ModelType(Enum):
    Continuous = "Continuous"
    Binary = "Binary"
    Multinomial = "Multinomial"


class FairnessTransformer(Transformer):
    """
    Fairness transformer

    Attributes:
        sensitive_attrs: List of columns containing sensitive attributes.
        target: Target variable to compute biases against.
        n_bins: Number of bins for sensitive attributes to be binned.
        target_n_bins: Number of bins for target to be binned, if None will use it as it is.
        positive_class: The sign of the biases depends on this class (positive biases have higher rate of this
            class). If not given, minority class will be used. Only used for binomial target variables.
        drop_dates: Whether to ignore sensitive attributes containing dates.
    """

    categorical_threshold_log_multiplier: float = 2.5
    min_num_unique: int = 10

    def __init__(
        self,
        sensitive_attrs: List[str],
        target: str,
        n_bins: int = 5,
        target_n_bins: Optional[int] = 5,
        positive_class: Optional[str] = None,
    ):
        super().__init__(name="fairness_transformer")

        self.sensitive_attrs = sensitive_attrs
        self.target = target
        self.n_bins = n_bins
        self.target_n_bins = target_n_bins
        self._used_columns = self.sensitive_attrs + [self.target]

        self._transformers: List[Transformer] = []
        self.models: Dict[str, ModelType] = dict()

    def fit(self, df: pd.DataFrame) -> "FairnessTransformer":
        df = self._get_dataframe_subset(df)

        if len(df) == 0:
            logger.warning("Empty DataFrame.")
            return self

        for c in df.columns:
            self.models[c] = self._infer_column_model(df[c])

        # Transformer for target column
        if self.models[self.target] == ModelType.Continuous and self.target_n_bins:
            df_target = df[[self.target]].copy()
            df_target = DTypeTransformer(self.target).fit_transform(df_target)
            binning_transformer = BinningTransformer(
                self.target, bins=self.target_n_bins, duplicates="drop", remove_outliers=0.1, include_lowest=True
            )
            df_target = binning_transformer.fit_transform(df_target)
            self._transformers.append(binning_transformer)

            self.models[self.target] = self._infer_column_model(df_target[self.target].astype(str))

        self._transformers.append(DTypeTransformer(self.target, out_dtype="str"))

        # Transformers for sensitive columns
        for col in self.sensitive_attrs:

            df = DTypeTransformer(col).fit_transform(df)

            transformer = self._get_sensitive_attr_transformer(df[col])
            if transformer:
                self._transformers.append(transformer)

            # We want to always convert to string otherwise grouping operations can fail
            to_str_transformer = DTypeTransformer(col, out_dtype="str")
            self._transformers.append(to_str_transformer)

        df = self.fit_transform(df)

        return super().fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self._get_dataframe_subset(df)

        if len(df) == 0:
            logger.warning("Empty DataFrame.")
            return df

        for t in self._transformers:
            df = t.transform(df)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        for t in self._transformers:
            t.fit(df)
            df = t.transform(df)

        return df

    def _get_sensitive_attr_transformer(self, column: pd.Series) -> Optional[Transformer]:
        column_name = column.name

        if self.models[column_name] == ModelType.Continuous:
            remove_outliers = None if column.dtype.kind == "M" else 0.1
            return BinningTransformer(
                column_name, bins=self.n_bins, remove_outliers=remove_outliers, duplicates="drop", include_lowest=True
            )

        return None

    def _get_dataframe_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in self._used_columns):
            raise KeyError("Target variable or sensitive attributes not present in DataFrame.")

        df = df[self._used_columns].copy()
        return df[~df[self.target].isna()]

    def _infer_column_model(self, column: pd.Series) -> "ModelType":
        num_rows = len(column)
        n_unique = column.nunique()

        if n_unique > max(self.min_num_unique, self.categorical_threshold_log_multiplier * np.log(num_rows)):
            return ModelType.Continuous

        elif n_unique == 2:
            return ModelType.Binary

        else:
            return ModelType.Multinomial
