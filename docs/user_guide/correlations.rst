Hidden Correlations
===================

Fairlens offers a range of correlation metrics that analyze associations between two
numerical series, two categorical series or between categorical and numerical ones.
These metrics are used both to generate correlation heatmaps of datasets to provide a general
overview and to detect columns that act as proxies for the sensitive attributes of datasets,
which pose the risk of training biased models.


Sensitive Proxy Detection
^^^^^^^^^^^^^^^^^^^^^^^^^

In some datasets, it is possible that some apparently insensitive attributes are correlated highly enough
with a sensitive column that they effectively become proxies for them and posing the danger to make a
biased machine learning model if the dataset is used for training.

As such, :code:`detection.py` provides utilities for scanning dataframes and detecting insensitive columns
that are correlated with a protected category. For a dataframe, a user can choose to scan the whole dataframe
and its columns or to provide an exterior Pandas series of interest that will be tested against the sensitive
columns of the data.

In a similar fashion to the detection function, users have the possibility to provide their own custom string
distance function and a threshold, as well as specify the correlation cutoff, which is a number representing
the minimum correlation coefficient needed to consider two columns to be correlated.

Let's first look at how we would go about detecting correlations inside a dataframe:

.. ipython:: python

    import pandas as pd

    columns = ["gender", "random", "score"]
    data = [["male", 10, 50], ["female", 20, 80], ["male", 20, 60], ["female", 10, 90]]

    df = pd.DataFrame(data, columns=columns)

Here the score seems to be correlated with gender, with females leaning towards somewhat higher scores.
This is picked up by the function, specifying both the insensitive and sensitive columns, as well as the
protected category of the sensitive one:

.. ipython:: python

    from fairlens.sensitive.correlation import find_sensitive_correlations

    find_sensitive_correlations(df)

In this example, the two scores are both correlated with sensitive columns, the first one with gender and
the second with nationality:

.. ipython:: python

    col_names = ["gender", "nationality", "random", "corr1", "corr2"]
    data = [
        ["woman", "spanish", 715, 10, 20],
        ["man", "spanish", 1008, 20, 20],
        ["man", "french", 932, 20, 10],
        ["woman", "french", 1300, 10, 10],
    ]
    df = pd.DataFrame(data, columns=col_names)

    find_sensitive_correlations(df)


Correlation Heatmaps
^^^^^^^^^^^^^^^^^^^^

The :code:`heatmap.py` module allows users to generate a correlation heatmap of any dataset by simply
passing the dataframe to the :code:`two_column_heatmap()` function, which will plot a heatmap from the
matrix of the correlation coefficients computed by using the Pearson Coefficient, the Kruskal-Wallis
Test and Cramer's V between each two of the columns (for numerical-numerical, categorical-numerical and
categorical-categorical associations, respectively).

To offer the possibility for extensibility, users are allowed to provide some or all of the correlation
metrics themselves instead of just using the defaults.

.. note::
    The :code:`fairlens.metrics` package provides a number of correlation metrics for any type of association.

    Alternatively, users can opt to implement their own metrics provided that they have two :code:`pandas.Series`
    objects as input and return a float that will be used as the correlation coefficient in the heatmap.

Let's look at an example by generating a correlation heatmap using the COMPAS dataset. First, we will load
the data and check what columns in contains.

.. ipython:: python

    df = pd.read_csv("../datasets/german_credit_data.csv")
    df

Now let us generate a heatmap using the default metrics first.

.. ipython:: python

    import matplotlib.pyplot as plt
    from fairlens.bias.heatmap import two_column_heatmap

    @verbatim
    two_column_heatmap(df)

    @verbatim
    plt.show()

Let's try generating a heatmap of the same dataset, but using some non-linear metrics
for numerical-numerical and numerical-categorical associations for added precision.

.. ipython:: python

    from fairlens.metrics import correlation as cm

    @verbatim
    two_column_heatmap(df, cm.distance_nn_correlation, cm.distance_cn_correlation, cm.cramers_v)

    @verbatim
    plt.show()
