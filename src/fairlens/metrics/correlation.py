"""
Collection of metrics, tests that measure the correlation between two univariate distributions.
"""

import warnings

import dcor as dcor
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def cramers_v(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that calculates the corrected Cramer's V statistic for categorical-categorical
    correlations, used in heatmap generation.

    Args:
        sr_a (pd.Series):
            First categorical series to analyze.
        sr_b (pd.Series):
            Second categorical series to analyze.

    Returns:
        float: Value of the statistic.
    """

    if len(sr_a.value_counts()) == 1:
        return 0
    if len(sr_b.value_counts()) == 1:
        return 0
    else:
        confusion_matrix = pd.crosstab(sr_a, sr_b)

        if confusion_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True

        chi2 = ss.chi2_contingency(confusion_matrix, correction=correct)[0]
        n = sum(confusion_matrix.sum())
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def pearson(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that calculates Pearson's correlation coefficent for numerical-numerical
    pairs of series, used in heatmap generation.

    Args:
        sr_a (pd.Series): First numerical series to analyze.
        sr_b (pd.Series): Second numerical series to analyze.

    Returns:
        float: Value of the coefficient.
    """
    return abs(sr_a.corr(sr_b))


def r2_mcfadden(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric used for categorical-numerical continuous. It trains two multinomial logistic
    regression models on the data, one using the numerical series as the feature and the other
    only using the intercept term as the input. The categorical column is used for the target
    labels. It then calculates the null and the model likelihoods based on them, which are
    used to compute the pseudo-R2 McFadden score, which is used as a correlation coefficient.

    Args:
        sr_a (pd.Series):
            The categorical series to analyze, representing target labels.
        sr_b (pd.Series):
            The numerical series to analyze.

    Returns:
        float: Value of the pseudo-R2 McFadden score.
    """
    x = sr_b.to_numpy().reshape(-1, 1)
    x = StandardScaler().fit_transform(x)
    y = sr_a.to_numpy()

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    lr_feature = linear_model.LogisticRegression()
    lr_feature.fit(x, y)

    y_one_hot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))

    log_pred = lr_feature.predict_log_proba(x)
    ll_feature = np.sum(y_one_hot * log_pred)

    lr_intercept = linear_model.LogisticRegression()
    lr_intercept.fit(np.ones_like(y).reshape(-1, 1), y)

    log_pred = lr_intercept.predict_log_proba(x)
    ll_intercept = np.sum(y_one_hot * log_pred)

    pseudo_r2 = 1 - ll_feature / ll_intercept

    return pseudo_r2


def kruskal_wallis(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that uses the Kruskal-Wallis H Test to obtain a p-value indicating the possibility
    that a categorical and numerical series are not correlated, used in heatmap generation.

    Args:
        sr_a (pd.Series):
            The categorical series to analyze, used for grouping the numerical one.
        sr_b (pd.Series):
            The numerical series to analyze.

    Returns:
        float:
            The correlation coefficient, calculating by subtracting the p-value from 1, as the
            p-value is the probability that the two columns are not correlated.
    """

    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    args = [group.array for group in arrays]
    try:
        _, p_val = ss.kruskal(*args, nan_policy="omit")
    except ValueError:
        return 0

    return p_val


def kruskal_wallis_boolean(sr_a: pd.Series, sr_b: pd.Series, p_cutoff: float = 0.1) -> bool:
    """Metric that uses the Kruskal-Wallis H Test to obtain a p-value that is used to determine
    whether the possibility that the columns obtained by grouping the continuous series
    by the categorical series come from the same distribution. Used for proxy detection.

    Args:
        sr_a (pd.Series):
            The categorical series to analyze, used for grouping the numerical one.
        sr_b (pd.Series):
            The numerical series to analyze.
        p_cutoff (float):
            The maximum admitted p-value for the distributions to be considered independent.

    Returns:
        bool: Bool value representing whether or not the two series are correlated.
    """

    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    if arrays:
        args = [np.array(group.array, dtype=float) for group in arrays]
        try:
            _, p_val = ss.kruskal(*args, nan_policy="omit")
        except ValueError:
            return False
        if p_val < p_cutoff:
            return True

    return False


def distance_nn_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that uses non-linear correlation distance to obtain a correlation coefficient for
    numerical-numerical column pairs.

    Args:
        sr_a (pd.Series):
            First numerical series to analyze.
        sr_b (pd.Series):
            Second numerical series to analyze.

    Returns:
        float:
            The correlation coefficient.
    """

    warnings.filterwarnings(action="ignore", category=UserWarning)

    if sr_a.size < sr_b.size:
        new_serie = pd.Series(sr_a.mean()).repeat(sr_b.size - sr_a.size)
        sr_a = pd.concat([sr_a, new_serie], ignore_index=True)
    elif sr_a.size > sr_b.size:
        new_serie = pd.Series(sr_b.mean()).repeat(sr_a.size - sr_b.size)
        sr_b = pd.concat([sr_b, new_serie], ignore_index=True)

    return dcor.distance_correlation(sr_a, sr_b)


def distance_cn_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric that uses non-linear correlation distance to obtain a correlation coefficient for
    categorical-numerical column pairs.

    Args:
        sr_a (pd.Series):
            The categorical series to analyze, used for grouping the numerical one.
        sr_b (pd.Series):
            The numerical series to analyze.

    Returns:
        float:
            The correlation coefficient.
    """

    warnings.filterwarnings(action="ignore", category=UserWarning)

    sr_a = sr_a.astype("category").cat.codes
    groups = sr_b.groupby(sr_a)
    arrays = [groups.get_group(category) for category in sr_a.unique()]

    total = 0.0
    n = len(arrays)

    for i in range(0, n):
        for j in range(i + 1, n):
            sr_i = arrays[i]
            sr_j = arrays[j]

            # Handle groups with a different number of elements.
            if sr_i.size < sr_j.size:
                new_serie = sr_i.sample(sr_j.size - sr_i.size, replace=True)
                sr_i = pd.concat([sr_i, new_serie], ignore_index=True)
            elif sr_i.size > sr_j.size:
                new_serie = sr_j.sample(sr_i.size - sr_j.size, replace=True)
                sr_j = pd.concat([sr_j, new_serie], ignore_index=True)
            total += dcor.distance_correlation(sr_i, sr_j)

    total /= n * (n - 1) / 2

    if total is None:
        return 0.0

    return total
