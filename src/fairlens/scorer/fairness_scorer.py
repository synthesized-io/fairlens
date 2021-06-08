import logging
from itertools import combinations

# from math import factorial
# from typing import Any, Dict, List, Optional, Sized, Tuple, Union
from typing import List, Optional

# import numpy as np
import pandas as pd

from ..bias import utils
from ..bias.metrics import stat_distance
from ..sensitive.detection import detect_names_df as detect

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
            detect_sensitive = True
            sensitive_attrs = []

        # Detect sensitive attributes
        if detect_sensitive:
            sensitive_attrs = list(set([k for (k, v) in detect(df).items() if v is not None]).union(sensitive_attrs))

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
    ) -> pd.DataFrame:
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

        df_pre = self.df

        if len(self.sensitive_attrs) == 0 or len(df_pre) == 0 or len(df_pre.dropna()) == 0:
            return 0.0, pd.DataFrame([], columns=["name", "target", "distance", "count"])

        max_comb = min(max_comb, len(self.sensitive_attrs)) if max_comb else len(self.sensitive_attrs)

        df_dists = []

        # Try all combinations of sensitive attributes
        for k in range(1, max_comb + 1):
            for sensitive_attr in combinations(self.sensitive_attrs, k):
                df_not_nan = df_pre[~(df_pre[list(sensitive_attr)] == "nan").any(axis=1)]
                if len(df_not_nan) == 0:
                    continue

                df_dist = self.calculate_distance(list(sensitive_attr), mode=mode, alpha=alpha)
                df_dists.append(df_dist)

        return pd.concat(df_dists, ignore_index=True)

    def calculate_distance(self, sensitive_attrs: List[str], mode: str = "auto", alpha: float = 0.05) -> pd.DataFrame:
        """Calculates the distance between the distribution of all the unique groups of values and the
        distribution without the respective value.

        Args:
            sensitive_attrs (List[str]):
                The list of sensitive attributes to consider.
            mode (str, optional):
                Choose a different metric to use. Defaults to automatically chosen metric depending on
                the distribution of the target variable.
            alpha (float, optional):
                Maximum p-value to accept a bias. Defaults to 0.05.

        Returns:
            pd.DataFrame:
                A dataframe consisting of the groups and their distances to the remaining dataset sorted
                in ascending order.
        """

        df = self.df
        target_attr = self.target_attr

        unique = df[sensitive_attrs].drop_duplicates()

        dist = []

        for _, row in unique.iterrows():
            sensitive_group = {attr: [value] for attr, value in row.to_dict().items()}

            pred = utils.get_predicates_mult(df, [sensitive_group])[0]

            distance = stat_distance(df, target_attr, df[pred][target_attr], df[~pred][target_attr], mode=mode)

            dist.append({
                "Group": ", ".join(row.to_dict().values()),
                "Distance": distance,
                "Proportion": len(df[pred]) / len(df)
            })

        return pd.DataFrame(dist)
