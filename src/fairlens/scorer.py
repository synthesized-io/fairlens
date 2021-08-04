"""
Automatically generate a fairness report for a dataset.
"""

import logging
from itertools import combinations
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

from . import utils

# from .bias.heatmap import two_column_heatmap
from .metrics.unified import stat_distance
from .plot.distr import mult_distr_plot

# from .sensitive.correlation import find_sensitive_correlations
from .sensitive.detection import detect_names_df

logger = logging.getLogger(__name__)


class FairnessScorer:
    """This class analyzes a given DataFrame, looks for biases and quantifies fairness."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_attr: str,
        sensitive_attrs: Optional[Sequence[str]] = None,
        detect_sensitive: bool = False,
        distr_type: Optional[str] = None,
        sensitive_distr_types: Optional[Mapping[str, str]] = None,
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
            distr_type (Optional[str], optional):
                The type of distribution of the target attribute. Can take values from
                ["categorical", "continuous", "binary", "datetime"]. If None, the type of
                distribution is inferred based on the data in the column. Defaults to None.
            sensitive_distr_types (Optional[Mapping[str, str]], optional):
                The type of distribution of the sensitive attributes. Passed as a mapping
                from sensitive attribute name to corresponding distribution type.
                Can take values from ["categorical", "continuous", "binary", "datetime"].
                If None, the type of distribution of all sensitive attributes are inferred
                based on the data in the respective columns. Defaults to None.
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
        self.sensitive_attrs = sorted(list(sensitive_attrs))

        # Infer the types of each distribution
        if distr_type is None:
            self.distr_type = utils.infer_distr_type(df[target_attr])
        else:
            self.distr_type = utils.DistrType(distr_type)

        t = sensitive_distr_types or {}
        self.sensitive_distr_types = [
            utils.DistrType(t[attr]) if attr in t else utils.infer_distr_type(df[attr]) for attr in self.sensitive_attrs
        ]

    def distribution_score(
        self,
        metric: str = "auto",
        method: str = "dist_to_rest",
        p_value: bool = False,
        max_comb: Optional[int] = None,
    ) -> pd.DataFrame:
        """Returns a dataframe consisting of all unique sub-groups and their statistical distance to the rest
        of the population w.r.t. the target variable.

        Args:
            metric (str, optional):
                Choose a metric to use. Defaults to automatically chosen metric depending on
                the distribution of the target variable.
            p_value (bool, optional):
                Whether or not to compute a p-value for the distances.
            max_comb (Optional[int], optional):
                Max number of combinations of sensitive attributes to be considered. Defaults to None.
        """

        df = self.df[self.sensitive_attrs + [self.target_attr]].copy()
        sensitive_attrs = self.sensitive_attrs

        # Bin continuous sensitive attributes
        for attr, distr_type in zip(self.sensitive_attrs, self.sensitive_distr_types):
            if distr_type.is_continuous() or distr_type.is_datetime():
                col = utils.infer_dtype(df[attr])
                df.loc[:, attr] = utils._bin_as_string(col, distr_type.value, prefix=True)

        # Convert binary attributes to 0s and 1s
        if self.distr_type.is_binary():
            df.loc[:, self.target_attr] = pd.factorize(df[self.target_attr])[0]

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

                df_dist = _calculate_distance(df, self.target_attr, list(sensitive_attr), metric, method, p_value)
                df_dists.append(df_dist)

        df_dist = pd.concat(df_dists, ignore_index=True)

        return df_dist.reset_index(drop=True)

    def plot_distributions(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        max_width: int = 3,
        max_bins: int = 8,
        show_hist: Optional[bool] = None,
        show_curve: Optional[bool] = None,
        shade: bool = True,
        normalize: bool = False,
        cmap: Optional[Sequence[Tuple[float, float, float]]] = None,
    ):
        """Plot the distributions of the target variable with respect to all sensitive values.

        Args:
            figsize (Optional[Tuple[int, int]], optional):
                The size of each figure if `separate` is True. Defaults to (6, 4).
            max_width (int, optional):
                The maximum amount of figures. Defaults to 3.
            show_hist (Optional[bool], optional):
                Shows the histogram if True. Defaults to True if the data is categorical or binary.
            show_curve (Optional[bool], optional):
                Shows a KDE if True. Defaults to True if the data is continuous or a date.
            shade (bool, optional):
                Shades the curve if True. Defaults to True.
            normalize (bool, optional):
                Normalizes the counts so the sum of the bar heights is 1. Defaults to False.
            cmap (Optional[Sequence[Tuple[float, float, float]]], optional):
                A sequence of RGB tuples used to colour the histograms. If None seaborn's default pallete
                will be used. Defaults to None.
            hist_kws (Optional[Mapping[str, Any]], optional):
                Additional keyword arguments passed to seaborn.histplot(). Defaults to None.
        """

        mult_distr_plot(
            self.df,
            self.target_attr,
            self.sensitive_attrs,
            figsize=figsize,
            max_width=max_width,
            max_bins=max_bins,
            show_hist=show_hist,
            show_curve=show_curve,
            shade=shade,
            normalize=normalize,
            cmap=cmap,
        )

    def demographic_report(
        self,
        metric: str = "auto",
        method: str = "dist_to_rest",
        alpha: Optional[float] = 0.95,
        min_count: Optional[int] = 100,
        max_rows: int = 10,
        hide_positive: bool = False,
    ):
        """Generate a report on the fairness of different groups of sensitive attributes.

        Args:
            metric (str, optional):
                Choose a custom metric to use. Defaults to automatically chosen metric depending on
                the distribution of the target variable.
            alpha (Optional[float], optional):
                Maximum p-value to accept a bias. Includes all sub-groups by default. Defaults to 0.95.
            min_count (Optional[int], optional):
                If set, sub-groups with less samples than min_count will be ignored. Defaults to 100.
            max_rows (int, optional):
                Maximum number of biased demographics to display. Defaults to 10.
            hide_positive (bool, optional):
                Hides positive distances if set to True. This may be useful when using metrics which can return
                negative distances (binomial distance), in order to inspect a skew in only one direction.
                Alternatively changing the method may yeild more significant results.
                Defaults to False.
        """

        df_dist = self.distribution_score(metric=metric, method=method, p_value=(alpha is not None))

        if alpha is not None:
            df_dist = df_dist[df_dist["P-Value"] < alpha]

        if min_count is not None:
            df_dist = df_dist[df_dist["Counts"] > min_count]

        score = calculate_score(df_dist)

        if hide_positive:
            df_dist = df_dist[df_dist["Distance"] < 0]

        print(f"Sensitive Attributes: {self.sensitive_attrs}")
        print("Most skewed demographics:")
        print(df_dist.sort_values("Distance", ascending=False, key=abs)[:max_rows].to_string(index=False))
        print(f"\nWeighted Mean Statistical Distance: {score}\n")

    def proxy_report(self):
        pass
        # TODO: Fix bugs in correlations
        # if detect_proxies:
        #     proxy_dict = find_sensitive_correlations(df)
        #     proxy_col = proxy_dict.keys()
        #     sensitive_col = [proxy_dict[key][0] for key in proxy_dict.keys()]
        #     category_col = [proxy_dict[key][1] for key in proxy_dict.keys()]

        #     df_proxy = pd.DataFrame(
        #         list(zip(proxy_col, sensitive_col, category_col)),
        #         columns=["Proxy", "Sensitive Column", "Protected Category"],
        #     )

        #     print(
        #         "Below are the detected proxies for sensitive attributes using a default \
        #             correlation coefficient cutoff of 0.75.\n"
        #     )
        #     print(df_proxy)

        # If full heatmap is set to true, the full correlation heatmap will be output. Otherwise, only the
        # correlations of the sensitive attributes with the non-sensitive ones will be calculated.
        # TODO: Fix heatmap formatting
        # if full_heatmap:
        #     two_column_heatmap(df)
        #     plt.show()
        # else:
        #     columns_sensitive = self.sensitive_attrs
        #     columns_nonsensitive = list(set(df.columns) - set(self.sensitive_attrs))

        #     two_column_heatmap(df, columns_x=columns_sensitive, columns_y=columns_nonsensitive)
        #     plt.show()


def calculate_score(df_dist: pd.DataFrame) -> float:
    """Calculate the weighted mean pairwise statistical distance.

    Args:
        df_dist (pd.DataFrame):
            A dataframe of statistical distances produced by or `fairlens.FairnessScorer.distribution_score`.

    Returns:
        float:
            The weighted mean statistical distance.
    """

    return (df_dist["Distance"].abs() * df_dist["Counts"]).sum() / df_dist["Counts"].sum()


def _calculate_distance(
    df: pd.DataFrame,
    target_attr: str,
    sensitive_attrs: Sequence[str],
    metric: str = "auto",
    method: str = "dist_to_rest",
    p_value: bool = False,
) -> pd.DataFrame:

    unique = df[sensitive_attrs].drop_duplicates()

    dist = []

    for _, row in unique.iterrows():
        sensitive_group = {attr: [value] for attr, value in row.to_dict().items()}

        pred = utils.get_predicates_mult(df, [sensitive_group])[0]

        if method == "dist_to_rest":
            pred_other = ~pred
        else:
            pred_other = pd.Series([True] * len(df))

        dist_res = stat_distance(df, target_attr, pred, pred_other, mode=metric, p_value=p_value)
        distance = dist_res[0]
        p = dist_res[1] if p_value else 0

        dist.append(
            {
                "Group": ", ".join(map(str, row.to_dict().values())),
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
