import logging
from abc import abstractmethod
from typing import List, Optional, TypeVar, Union

import pandas as pd

from .exceptions import TransformerNotFitError

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

    def __init__(self, name: str, dtypes: Optional[List[str]] = None):
        super().__init__()
        self.name = name
        self._fitted = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

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

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise TransformerNotFitError("Transformer not fitted yet, please call 'fit()' before calling transform.")
