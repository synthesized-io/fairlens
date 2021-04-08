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

    def __eq__(self, other):
        def attrs(x):
            return dict(filter(lambda x: not x[0].startswith("_"), x.__dict__.items()))

        try:
            np.testing.assert_equal(attrs(self), attrs(other))
            return True
        except AssertionError:
            return False

    def __call__(self, x: pd.DataFrame, inverse=False, **kwargs) -> pd.DataFrame:
        self._assert_fitted()
        if not inverse:
            return self.transform(x, **kwargs)
        else:
            return self.inverse_transform(x, **kwargs)

    def fit(self: TransformerType, x: Union[pd.Series, pd.DataFrame]) -> TransformerType:
        if not self._fitted:
            self._fitted = True
        return self

    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fit(df).transform(df)
        return df

    def _assert_fitted(self):
        if not self._fitted:
            raise TransformerNotFitError("Transformer not fitted yet, please call 'fit()' before calling transform.")

    @property
    def in_columns(self) -> List[str]:
        return [self.name]

    @property
    def out_columns(self) -> List[str]:
        return [self.name]


class SequentialTransformer(Transformer, MutableSequence[Transformer]):
    """
    Transform data using a sequence of pre-defined Transformers.
    Each transformer can act on different columns of a data frame,
    or the same column. In the latter case, each transformer in
    the sequence is fit to the transformed data from the previous.
    """

    def __init__(self, name: str, transformers: Optional[List[Transformer]] = None, dtypes: Optional[List] = None):
        super().__init__(name, dtypes)

        if transformers is None:
            self._transformers: List[Transformer] = []
        else:
            self._transformers = transformers

    def insert(self, idx: int, o: Transformer) -> None:
        self._transformers.insert(idx, o)

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}", dtypes={self.dtypes}, transformers={self._transformers})'

    def __getitem__(self, key: int):
        return self._transformers[key]

    def __setitem__(self, idx, o) -> None:
        self._transformers[idx] = o

    def __delitem__(self, idx: Union[int, slice]) -> None:
        del self._transformers[idx]

    def __iter__(self) -> Iterator[Transformer]:
        yield from self._transformers

    def __len__(self) -> int:
        return len(self._transformers)

    def __add__(self, other: "BagOfTransformers") -> "BagOfTransformers":
        return SequentialTransformer(name=self.name, transformers=self._transformers + other._transformers)

    def __contains__(self, key: object) -> bool:
        assert isinstance(key, str)
        return True if key in [t.name for t in self._transformers] else False

    def __reversed__(self):
        return reversed(self._transformers)

    def fit(self, df: pd.DataFrame) -> "BagOfTransformers":
        df = df.copy()  # have to copy because Transformer.transform modifies df

        for transformer in self:
            transformer.fit(df)
            df = transformer.transform(df)

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self._assert_fitted()

        df = df.copy()
        for transformer in self:
            df = transformer.transform(df, **kwargs)

        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        for transformer in reversed(self):
            df = transformer.inverse_transform(df, **kwargs)

        return df

    @property
    def in_columns(self) -> List[str]:
        return list(set([column for transformer in self for column in transformer.in_columns]))

    @property
    def out_columns(self) -> List[str]:
        return list(set([column for transformer in self for column in transformer.out_columns]))
