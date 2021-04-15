import logging
from abc import abstractmethod
from typing import (Any, Callable, Collection, Dict, Iterator, List, MutableSequence, Optional, Tuple, Type, TypeVar,
                    Union)

import numpy as np
import pandas as pd

from .exceptions import NonInvertibleTransformError, TransformerNotFitError

logger = logging.getLogger(__name__)

TransformerType = TypeVar("TransformerType", bound="Transformer")


class Transformer:
    """
    Base class for data frame transformers.
    Derived classes must implement transform. The
    fit method is optional, and should be used to
    extract required transform parameters from the data.
    Attributes:
        name: the data frame column to transform.
        dtypes: list of valid dtypes for this
          transformation, defaults to None.
    """

    def __init__(self, name: str, dtypes: Optional[List] = None):
        super().__init__()
        self.name = name
        self.dtypes = dtypes
        self._fitted = False

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, dtypes={self.dtypes})"

    def fit(self: TransformerType, x: Union[pd.Series, pd.DataFrame]) -> TransformerType:
        if not self._fitted:
            self._fitted = True
        return self

    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fit(df).transform(df)
        return df

    def _assert_fitted(self):
        if not self._fitted:
            raise TransformerNotFitError("Transformer not fitted yet, please call 'fit()' before calling transform.")

