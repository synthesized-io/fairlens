Detecting sensitive attributes
==============================

Fairlens contains tools that allow users to analyse their datasets in order to detect columns that are
sensitive or that act as proxies for other protected attributes, based on customisable configurations in
its :code:`sensitive` module.


Sensitive Attribute Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Detecting sensitive columns in a dataframe is the main functionality of :code:`detection.py`. This is done
by using :code:`detect_names_df()`, a function which finds sensitive columns and builds a dictionary
mapping the attribute names to the corresponding protected category (read from the configuration).
Let us take a look at an example dataframe, based on the default configuration :code:`config_engb.json`:

.. ipython:: python

    import pandas as pd

    columns = ["native", "location", "house", "subscription", "salary", "religion", "score"]
    df = pd.DataFrame(columns=columns)

In this scenario, we can use the function to get:

.. ipython:: python

    from fairlens.sensitive import detection as dt

    dt.detect_names_df(df)

In some cases, the names of the dataframe columns alone might not be conclusive enough to decide on
the sensitivity. In those cases, :code:`detect_names_df()` has the option of enabling the
:code:`deep_search` flag to look at the actual data entries. For example, let's assume we have a
dataframe containing data referring to protected attributes, but where the column names are not
related and let's try using detection as in the previous example:

.. ipython:: python

    columns = ["A", "B", "C", "Salary", "D", "Score"]
    data = [
        ["male", "hearing impairment", "heterosexual", "50000", "christianity", "10"],
        ["female", "obsessive compulsive disorder", "asexual", "60000", "daoism", "10"],
    ]
    df = pd.DataFrame(data, columns=columns)

    dt.detect_names_df(df)

As we can see, since the column names do not have a lot of meaning, shallow search will not suffice.
However, if we turn :code:`deep_search` on:

.. ipython:: python

    dt.detect_names_df(df, deep_search=True)

It is also possible for users to implement their own string distance functions to be used by the
detection algorithm. By default, Ratcliff-Obershelp algorithm is used, but any function with type
:code:`Callable[[Optional[str], Optional[str]], float]` can be used. The detection threshold can
also be changed to modify the strictness of the fuzzy matching.

Custom Configurations
^^^^^^^^^^^^^^^^^^^^^

The sensitive or protected group attribute detection algorithm is based on an underlying configuration, which is
a JSON file containing the sensitive categories, each having a list of synonyms and possible values attached to them.
Since currently the detection algorithm is based on fuzzy string matching, different languages and scopes will require
new comprehensive configurations.

The default configuration is in the English language and in accordance with the UK Government's official protected group
and category list. The configuration can be changed through API functions from :code:`detection.py`. For example, in order
to change the it to a new configuration :code:`config_custom.json` placed is the :code:`configs` folder from the
:code:`sensitive` module:

.. ipython:: python
    :verbatim:

    from fairlens.sensitive import detection as dt

    dt.change_config("./configs/config_custom.json")

Any new operations performed on dataframes using functions from :code:`detection.py` will assume that the contents of the new
configuration are the objects of interest and use them for inference.


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
