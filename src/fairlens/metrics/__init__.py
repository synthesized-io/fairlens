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

__all__ = [
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
    "auto_distance",
    "correlation_matrix",
    "stat_distance",
]
