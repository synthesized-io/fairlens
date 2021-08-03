"""
Collection of distance, correlation metrics.
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

from .significance import (  # isort:skip
    binom_test,
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
    "binom_test",
    "binominal_proportion_p_value",
    "binominal_proportion_interval",
    "bootstrap_binned_statistic",
    "bootstrap_statistic",
    "permutation_statistic",
    "resampling_p_value",
    "resampling_interval",
]
