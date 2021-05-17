import logging

# from math import factorial
# from typing import Any, Dict, List, Optional, Sized, Tuple, Union
from typing import List, Optional, Tuple, Union

# import numpy as np
import pandas as pd

# from ..bias import utils
from ..bias.metrics import ks_distance

# from itertools import combinations


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
) -> Tuple[float, pd.DataFrame]:
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

    pass


#     df_pre = self.df

#     if len(self.sensitive_attrs) == 0 or len(df_pre) == 0 or len(df_pre.dropna()) == 0:
#         return 0.0, pd.DataFrame([], columns=["name", "target", "distance", "count"])

#     biases = []
#     max_comb = min(max_comb, len(self.sensitive_attrs)) if max_comb else len(self.sensitive_attrs)

#     # Try all combinations of sensitive attributes
#     for k in range(1, max_comb + 1):
#         for sensitive_attr in combinations(self.sensitive_attrs, k):
#             df_not_nan = df_pre[~(df_pre[list(sensitive_attr)] == "nan").any(axis=1)]
#             if len(df_not_nan) == 0:
#                 continue

#             df_dist = calculate_distance(df_not_nan, list(sensitive_attr), self.target_attr, mode=mode, alpha=alpha)
#             biases.extend(self.format_bias(df_dist))

#             n += 1

#     return None


def calculate_distance(
    df: pd.DataFrame, sensitive_attrs: List[str], target_attr: str, mode: str = "auto", alpha: float = 0.05
) -> pd.DataFrame:
    """Calculates the distance between the distribution of the target attribute with respect to the group
    and the remaining data points.

    Args:
        df (pd.DataFrame):
            The input dataframe.
        sensitive_attrs (List[str]):
            The list of sensitive attributes to consider.
        target_attr (str):
            The target attribute.
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

    unique = df[sensitive_attrs].drop_duplicates()

    dist = []

    for _, row in unique.iterrows():
        sensitive_group = {attr: [value] for attr, value in row.to_dict().items()}

        distance: Union[float, Tuple[float, float]] = ks_distance(df, target_attr, sensitive_group)

        if isinstance(distance, tuple):
            distance, pval = distance

        dist.append({", ".join(row.to_dict().keys()): ", ".join(row.to_dict().values()), "Distance": distance})

    return pd.DataFrame(dist).sort_values("Distance")
