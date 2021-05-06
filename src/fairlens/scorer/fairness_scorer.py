import logging

# from itertools import combinations
# from math import factorial
# from typing import Any, Dict, List, Optional, Sized, Tuple, Union
from typing import List, Optional

# import numpy as np
import pandas as pd
from tqdm import tqdm

# from ..bias import utils
# from ..sensitive.detection import detect_names_df

logger = logging.getLogger(__name__)


class FairnessScorer:
    """This class analyzes a given DataFrame, looks for biases and quantifies its fairness."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_attr: str,
        sensitive_attrs: Optional[List[str]] = None,
        detect_sensitive: bool = False,
        detect_hidden: bool = False,
    ):
        """Fairness Scorer constructor

        Args:
            df (pd.DataFrame):
                Input DataFrame to be scored.
            target_attr (str):
                The target attribute name.
            sensitive_attrs (Optional[List[str]], optional):
                The sensitive attribute names. Defaults to None.
            detect_sensitive (bool, optional):
                Whether to try to detect sensitive attributes from the column names. Defaults to False.
            detect_hidden (bool, optional):
                Whether to try to detect sensitive attributes from hidden correlations with other sensitive
                attributes. Defaults to False.
        """

        if sensitive_attrs is None:
            if detect_sensitive is False:
                raise ValueError("If no 'sensitive_attr' is given, 'detect_sensitive' must be set to True.")

            sensitive_attrs = []

        # Detect sensitive attributes here

        if len(sensitive_attrs) == 0:
            logger.warning("No sensitive attributes detected. Fairness score will always be 0.")

        self.df = df
        self.target_attr = target_attr
        self.sensitive_attrs = sensitive_attrs

    def distribution_score(
        self,
        mode: str = "auto",
        alpha: float = 0.05,
        min_dist: Optional[float] = None,
        min_count: Optional[int] = 50,
        weighted: bool = True,
        max_comb: Optional[int] = 3,
        condense_output: bool = True,
    ):
        """Returns the biases and fairness score by analyzing the distribution difference between sensitive
        variables and the target variable.

        Args:
            mode (str, optional):
                Choose a different metric to use. Defaults to automatically chosen metric depending on
                the distribution of the target variable.
            alpha (float, optional):
                Maximum p-value to accept a bias. Defaults to 0.05.
            min_dist (Optional[float], optional):
                If set, any bias with smaller distance than min_dist will be ignored. Defaults to None.
            min_count (Optional[int], optional):
                If set, any bias with less samples than min_count will be ignored. Defaults to 50.
            weighted (bool, optional):
                Whether to weight the average of biases on the size of each sample. Defaults to True.
            max_comb (Optional[int], optional):
                Max number of combinations of sensitive attributes to be considered. Defaults to 3.
            condense_output (bool, optional):
                Whether to return one row per group or one per group and target. Defaults to True.
        """

        pbar = tqdm(total=100)

        if len(self.sensitive_attrs) == 0 or len(self.df) == 0 or len(self.df.dropna()) == 0:
            pbar.update(0)
            pbar.update(100)
            return 1.0, pd.DataFrame([], columns=["name", "target", "distance", "count"])

        # biases = []
        # max_comb = min(max_comb, len(self.sensitive_attrs)) if max_comb else len(self.sensitive_attrs)

        # Try all combinations of sensitive attributes
