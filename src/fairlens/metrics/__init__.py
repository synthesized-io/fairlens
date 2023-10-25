"""
Collection of distance, correlation metrics and statistical tests.
"""

from .unified import auto_distance, correlation_matrix, stat_distance

from .distance import (  # isort:skip
    DistanceMetric,
    CategoricalDistanceMetric,
    ContinuousDistanceMetric,
    BinomialDistance,
    MeanDistance,
    KolmogorovSmirnovDistance,
    KruskalWallis,
    EarthMoversDistance,
    KullbackLeiblerDivergence,
    JensenShannonDivergence,
    Norm,
    HellingerDistance,
)

from .correlation import (  # isort:skip
    cramers_v,
    distance_cn_correlation,
    distance_nn_correlation,
    r2_mcfadden,
    kruskal_wallis,
    kruskal_wallis_boolean,
)

from .significance import (  # isort:skip
    binomtest,
    binominal_proportion_p_value,
    binominal_proportion_interval,
    bootstrap_binned_statistic,
    bootstrap_statistic,
    permutation_statistic,
    resampling_p_value,
    resampling_interval,
)

__all__ = [
    "auto_distance",
    "correlation_matrix",
    "stat_distance",
    "DistanceMetric",
    "CategoricalDistanceMetric",
    "ContinuousDistanceMetric",
    "BinomialDistance",
    "MeanDistance",
    "KolmogorovSmirnovDistance",
    "KruskalWallis",
    "EarthMoversDistance",
    "KullbackLeiblerDivergence",
    "JensenShannonDivergence",
    "Norm",
    "HellingerDistance",
    "cramers_v",
    "distance_cn_correlation",
    "distance_nn_correlation",
    "r2_mcfadden",
    "kruskal_wallis",
    "kruskal_wallis_boolean",
    "binomtest",
    "binominal_proportion_p_value",
    "binominal_proportion_interval",
    "bootstrap_binned_statistic",
    "bootstrap_statistic",
    "permutation_statistic",
    "resampling_p_value",
    "resampling_interval",
]
