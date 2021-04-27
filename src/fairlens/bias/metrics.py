from typing import List

import numpy as np
import pandas as pd
from pyemd import emd as pemd

# from scipy.stats import ks_2samp


def compute_probabilities(df: pd.DataFrame, sensitive_attrs: List[str], target_attr: str) -> pd.DataFrame:
    """Computes the probability distributions for all sensitive groups of the given sensitive attributes
    with respect to the target attribute

    Args:
        df (pd.DataFrame):
            The input dataframe.
        sensitive_attrs (List[str]):
            The sensitive attributes to be considered.
        target_attr (str):
            The target attribute with respect to which the probabilities will be computed.

    Returns:
        pd.DataFrame:
            A dataframe containing the sensitive groups and their counts, probabilities
    """

    # Get group counts & rates
    sensitive_group_target_counts = df.groupby(sensitive_attrs + [target_attr])[target_attr].aggregate(Count="count")
    sensitive_group_size = df.groupby(sensitive_attrs).size()
    sensitive_group_target_counts["Rate"] = sensitive_group_target_counts["Count"] / sensitive_group_size

    # Get total counts & rates
    target_totals = df.groupby(target_attr)[target_attr].aggregate(Count="count")
    target_totals["Rate"] = target_totals / len(df)
    target_totals = target_totals.set_index(pd.MultiIndex.from_tuples([("Total", a) for a in target_totals.index]))

    return pd.concat((sensitive_group_target_counts, target_totals))


def emd(df: pd.DataFrame, sensitive_attrs: List[str], target_attr: str) -> pd.DataFrame:
    """Computes the Earth Mover's Distance for all sensitive groups of the given sensitive attributes
    with respect to the target attribute

    Args:
        df (pd.DataFrame):
            The input dataframe.
        sensitive_attrs (List[str]):
            The sensitive attributes to consider.

    Returns:
        pd.DataFrame:
            A dataframe containing the sensitive groups and their counts, Earth Mover's Distance.
    """

    df_count = compute_probabilities(df, sensitive_attrs, target_attr)

    emd_dist = []
    space = df_count.index.get_level_values(1).unique()

    for sensitive_value in df_count.index.get_level_values(0).unique():
        p_counts = df_count["Count"][sensitive_value].to_dict()

        # Remove counts in current subsample
        q_counts = {k: v - p_counts.get(k, 0) for k, v in df_count["Count"]["Total"].to_dict().items()}

        p = np.array([float(p_counts[x]) if x in p_counts else 0.0 for x in space])
        q = np.array([float(q_counts[x]) if x in q_counts else 0.0 for x in space])

        p /= np.sum(p)
        q /= np.sum(q)

        distance_space = 1 - np.eye(len(space))

        emd_dist.append(
            {
                df_count.index.names[0]: sensitive_value,
                "Distance": pemd(p, q, distance_space),
                "Count": df_count["Count"][sensitive_value].sum(),
            }
        )

    return pd.DataFrame(emd_dist).set_index(df_count.index.names[0])


# def ks_distance(self, df: pd.DataFrame, sensitive_attr: List[str], alpha: float = 0.05) -> pd.DataFrame:
#     # ignore rows which have nans in any of the given sensitive attrs
#     groups = df.groupby(sensitive_attr).groups
#     distances = []
#     for sensitive_attr_values, idxs in groups.items():
#         target_group = df.loc[df.index.isin(idxs), self.target]
#         target_rest = df.loc[~df.index.isin(idxs), self.target]
#         if len(target_group) == 0 or len(target_rest) == 0:
#             continue

#         dist, pval = ks_2samp(target_group, target_rest)
#         if pval < alpha:
#             if np.mean(target_group) < df[self.target].mean():
#                 dist = -dist

#             if isinstance(sensitive_attr_values, tuple) and len(sensitive_attr_values) > 1:
#                 sensitive_attr_str = "({})".format(", ".join([str(sa) for sa in sensitive_attr_values]))
#                 sensitive_attr_values = list(sensitive_attr_values)
#             else:
#                 sensitive_attr_str = sensitive_attr_values
#                 sensitive_attr_values = [sensitive_attr_values]

#             self.values_str_to_list[sensitive_attr_str] = sensitive_attr_values
#             distances.append([sensitive_attr_str, len(idxs), dist])

#     name = sensitive_attr_concat_name(sensitive_attr)
#     self.names_str_to_list[name] = sensitive_attr

#     return pd.DataFrame(distances, columns=[name, "Count", "Distance"]).set_index(name)
