import logging
from itertools import combinations
from typing import Optional, Sequence, Tuple

import pandas as pd

from ..bias import utils
from ..bias.metrics import stat_distance
from ..sensitive.detection import detect_names_df

logger = logging.getLogger(__name__)


class FairnessScorer:
    """This class analyzes a given DataFrame, looks for biases and quantifies fairness."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_attr: str,
        sensitive_attrs: Optional[Sequence[str]] = None,
        detect_sensitive: bool = False,
        detect_hidden: bool = False,
    ):
        """Fairness Scorer constructor

        Args:
            df (pd.DataFrame):
                Input DataFrame to be scored.
            target_attr (str):
                The target attribute name.
            sensitive_attrs (Optional[Sequence[str]], optional):
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
            attr_dict = detect_names_df(df, deep_search=True).items()
            sensitive_attrs = list(set([k for (k, v) in attr_dict if v is not None]).union(sensitive_attrs))

        if len(sensitive_attrs) == 0:
            logger.warning("No sensitive attributes detected. Fairness score will always be 0.")

        self.df = df
        self.target_attr = target_attr
        self.sensitive_attrs = list(sensitive_attrs)

    def distribution_score(
        self,
        mode: str = "auto",
        alpha: Optional[float] = None,
        min_prop: Optional[float] = 0.1,
        max_prop_thresh: Optional[float] = 0.95,
        min_dist: Optional[float] = None,
        min_count: Optional[int] = None,
        max_comb: Optional[int] = None,
    ) -> Tuple[float, pd.DataFrame]:
        """Returns an overall bias score and dataframe consisting of the biased sub-groups by analyzing the
        difference in distribution between sensitive-subgroups and the data.

        Args:
            mode (str, optional):
                Choose a different metric to use. Defaults to automatically chosen metric depending on
                the distribution of the target variable.
            alpha (Optional[float], optional):
                Maximum p-value to accept a bias. Includes all sub-groups by default. Defaults to None.
            min_prop (Optional[int], optional):
                If set, sub-groups with sample sizes representing a smaller proportion of the data than
                min_prop will be ignored. Defaults to 0.1.
            max_prop_thresh (Optional[int], optional):
                If set, sensitive attributes with a single subgroup representing a larger proportion of the data
                than max_prop_thresh will be ignored. Defaults to 0.95.
            min_count (Optional[int], optional):
                If set, sub-groups with less samples than min_count will be ignored. Defaults to None.
            min_dist (Optional[float], optional):
                If set, sub-groups with a smaller distance score than min_dist will be ignored.
                Defaults to None.
            max_comb (Optional[int], optional):
                Max number of combinations of sensitive attributes to be considered. Defaults to None.
        """

        df = self.df

        # Ignore sensitive attributes that have overly concentrated values. (Room for improvement)
        if max_prop_thresh is not None:
            sensitive_attrs = [
                s for s in self.sensitive_attrs if df[s].value_counts().max() < max_prop_thresh * len(df)
            ]

        if len(sensitive_attrs) == 0 or len(df) == 0 or len(df.dropna()) == 0:
            return 0.0, pd.DataFrame([], columns=["Group", "Distance", "Proportion", "Counts"])

        max_comb = min(max_comb, len(sensitive_attrs)) if max_comb is not None else len(sensitive_attrs)
        df_dists = []

        # Try all combinations of sensitive attributes
        for k in range(1, max_comb + 1):
            for sensitive_attr in combinations(sensitive_attrs, k):
                df_not_nan = df[~(df[list(sensitive_attr)] == "nan").any(axis=1)]
                if len(df_not_nan) == 0:
                    continue

                df_dist = self.calculate_distance(list(sensitive_attr), mode=mode, p_value=(alpha is not None))
                df_dists.append(df_dist)

        df_dist = pd.concat(df_dists, ignore_index=True)

        if alpha is not None:
            df_dist = df_dist[df_dist["P-Value"] < alpha]

        if min_prop is not None:
            df_dist = df_dist[df_dist["Counts"] > (min_prop * len(df))]

        if min_count is not None:
            df_dist = df_dist[df_dist["Counts"] > min_count]

        if min_dist is not None:
            df_dist = df_dist[df_dist["Distance"] > min_dist]

        score = (df_dist["Distance"] * df_dist["Counts"]).sum() / df_dist["Counts"].sum()

        return score, df_dist.reset_index(drop=True)

    def calculate_distance(
        self, sensitive_attrs: Sequence[str], mode: str = "auto", p_value: bool = False
    ) -> pd.DataFrame:
        """Calculates the distance between the distribution of all the unique groups of values and the
        distribution without the respective value.

        Args:
            sensitive_attrs (Sequence[str]):
                The list of sensitive attributes to consider.
            mode (str, optional):
                Choose a different metric to use. Defaults to automatically chosen metric depending on
                the distribution of the target variable.
            p_value (bool, optional):
                Whether or not to compute a p-value. Defaults to False.

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

            group = df[pred][target_attr]
            remaining = df[~pred][target_attr]

            dist_res = stat_distance(df, target_attr, group, remaining, mode=mode, p_value=p_value)
            distance = dist_res[0]
            p = dist_res[1] if p_value else 0

            dist.append(
                {
                    "Group": ", ".join(row.to_dict().values()),
                    "Distance": distance,
                    "Proportion": len(df[pred]) / len(df),
                    "Counts": len(df[pred]),
                    "P-Value": p,
                }
            )

        df_dist = pd.DataFrame(dist)

        if not p_value:
            df_dist.drop(columns=["P-Value"], inplace=True)

        return df_dist
