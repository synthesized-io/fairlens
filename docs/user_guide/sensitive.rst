Sensitive Attribute Detection
=============================

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
    import fairlens as fl

    columns = ["native", "location", "house", "subscription", "salary", "religion", "score"]
    df = pd.DataFrame(columns=columns)

In this scenario, we can use the function to get:

.. ipython:: python

    fl.sensitive.detect_names_df(df)

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

    fl.sensitive.detect_names_df(df)

As we can see, since the column names do not have a lot of meaning, shallow search will not suffice.
However, if we turn :code:`deep_search` on:

.. ipython:: python

    fl.sensitive.detect_names_df(df, deep_search=True)

It is also possible for users to implement their own string distance functions to be used by the
detection algorithm. By default, Ratcliff-Obershelp algorithm is used, but any function with type
:code:`Callable[[Optional[str], Optional[str]], float]` can be used. The detection threshold can
also be changed to modify the strictness of the fuzzy matching.

Let us try applying the detection functionality in a more practical scenario, using the COMPAS
dataset:

.. ipython:: python

    df = pd.read_csv("../datasets/compas.csv")
    df.head()

    # Apply shallow detection algorithm.
    fl.sensitive.detect_names_df(df)

As we can see, the sensitive categories from the dataframe have been picked out by the shallow search.
Let's now see what happens when we deep search, but just to make the task a bit more difficult, let's rename
the sensitive columns to have random names.

.. ipython:: python

    df_deep = pd.read_csv("../datasets/compas.csv")
    df_deep = df_deep.rename(columns={"Ethnicity": "A", "Language": "Random", "MaritalStatus": "B", "Sex": "C"})

    # Apply deep detection algorithm.
    fl.sensitive.detect_names_df(df, deep_search=True)

The same sensitive columns have been picked, but based solely on their content, as the column names themselves have
become non-sugestive.

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
