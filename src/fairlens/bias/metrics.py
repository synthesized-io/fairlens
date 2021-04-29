from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyemd import emd as pemd
from scipy.stats import entropy, ks_2samp

from . import utils


def class_imbalance(
    df: pd.DataFrame, target_attr: str, group1: Dict[str, List[Any]], group2: Optional[Dict[str, List[Any]]] = None
) -> float:

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    return (df[pred1].nunique() - df[pred2].nunique()) / df[target_attr].nunique()


def emd(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Optional[Dict[str, List[Any]]] = None,
    group1_counts: Optional[Dict[Any, int]] = None,
    group2_counts: Optional[Dict[Any, int]] = None,
) -> float:

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    group1_counts = group1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()
    group2_counts = group2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += group1_counts.get(val, 0)
        q[i] += group2_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()

    distance_space = 1 - np.eye(len(space))

    return pemd(p, q, distance_space)


def ks_distance(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Optional[Dict[str, List[Any]]] = None,
    alpha: float = 0.05,
) -> Tuple[float, float]:

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    return ks_2samp(df[pred1][target_attr], df[pred2][target_attr])


def kl_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Optional[Dict[str, List[Any]]] = None,
    group1_counts: Optional[Dict[Any, int]] = None,
    group2_counts: Optional[Dict[Any, int]] = None,
) -> float:

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    group1_counts = group1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()
    group2_counts = group2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += group1_counts.get(val, 0)
        q[i] += group2_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()

    return entropy(p, q)


def js_divergence(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Optional[Dict[str, List[Any]]] = None,
    group1_counts: Optional[Dict[Any, int]] = None,
    group2_counts: Optional[Dict[Any, int]] = None,
    total_counts: Optional[Dict[Any, int]] = None,
) -> float:

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    group1_counts = group1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()
    group2_counts = group2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()
    total_counts = total_counts or df.groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    pq = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += group1_counts.get(val, 0)
        q[i] += group2_counts.get(val, 0)
        pq[i] += total_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()
    pq /= pq.sum()

    return (entropy(p, pq) + entropy(q, pq)) / 2


def lp_norm(
    df: pd.DataFrame,
    target_attr: str,
    group1: Dict[str, List[Any]],
    group2: Optional[Dict[str, List[Any]]] = None,
    p=2,
    group1_counts: Optional[Dict[Any, int]] = None,
    group2_counts: Optional[Dict[Any, int]] = None,
) -> float:

    pred1, pred2 = utils.get_predicates(df, group1, group2)

    group1_counts = group1_counts or df[pred1].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()
    group2_counts = group2_counts or df[pred2].groupby(target_attr)[target_attr].aggregate(Count="count").to_dict()

    space = df[target_attr].unique()

    p = np.zeros(len(space))
    q = np.zeros(len(space))
    for i, val in enumerate(space):
        p[i] += group1_counts.get(val, 0)
        q[i] += group2_counts.get(val, 0)

    p /= p.sum()
    q /= q.sum()

    return np.linalg.norm(p, ord=p)
