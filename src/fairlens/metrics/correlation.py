"""
Collection of metrics that measure the correlation between two distributions.
"""

import warnings

import dcor as dcor
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def r2_linear_correlation(sr_a: pd.Series, sr_b: pd.Series) -> float:
    """Metric used for categorical-numerical continuous. It trains a linear model on
    a percentage of the data, with the numerical series elements as inputs and classes
    from the categorical series as targets. It is then tested on the remainder of the
    series and the predictions alongside the true values are used to compute a R2 score.

    Args:
        sr_a (pd.Series):
            The categorical series to analyze, representing target classes.
        sr_b (pd.Series):
            The numerical series to analyze.

    Returns:
        float: Value of the R2 score.
    """
    sr_a = sr_a.apply(pd.to_numeric, errors="coerce")

    x = sr_b.to_numpy()
    y_categorical = sr_a.to_numpy()

    enc = LabelEncoder()
    enc.fit(y_categorical)
    y = enc.transform(y_categorical)
    x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=0.2)

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    return r2_score(y_true=y_test, y_pred=y_pred)


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
        sr_a = sr_a.append(pd.Series(sr_a.mean()).repeat(sr_b.size - sr_a.size), ignore_index=True)
    elif sr_a.size > sr_b.size:
        sr_b = sr_b.append(pd.Series(sr_b.mean()).repeat(sr_a.size - sr_b.size), ignore_index=True)

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
                sr_i = sr_i.append(sr_i.sample(sr_j.size - sr_i.size, replace=True), ignore_index=True)
            elif sr_i.size > sr_j.size:
                sr_j = sr_j.append(sr_j.sample(sr_i.size - sr_j.size, replace=True), ignore_index=True)
            total += dcor.distance_correlation(sr_i, sr_j)

    total /= n * (n - 1) / 2

    if total is None:
        return 0.0

    return total
