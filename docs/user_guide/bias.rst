Measuring bias and fairness
===========================

Fairlens allows users to make use of a wide range of statistical distance metrics to measure the difference
between the distributions of two potentially sensitive sub-groups of data. In addition users can use the
:code:`FairnessScorer` to automatically assess the fairness of columns in a dataset with respect to a target column.


Fairness Scorer
^^^^^^^^^^^^^^^

Let's test out the fairlens scorer on the compas dataset.

First we will import the required packages and load the compas dataset.

.. ipython:: python

  import pandas as pd

  df = pd.read_csv("../datasets/compas.csv")
  df.head()

.. ipython:: python
  :verbatim:

  sensitive_attrs = ["Ethnicity", "Sex"]
  target_attr = "RawScore"

  fscorer = FairnessScorer(df, target_attr, sensitive_attrs)
  fscorer.distribution_score()


Statistical Distances
^^^^^^^^^^^^^^^^^^^^^

Fairlens supports multiple distance metrics which you can use via the method :code:`stat_distance`.
Let's import this method.

.. ipython:: python

  from fairlens.bias.metrics import stat_distance

We need to define which groups of data we want to measure the distance between and the target attribute.

.. ipython:: python

  group1 = {"Ethnicity": ["African-American"]}
  group2 = {"Ethnicity": ["Caucasian"]}
  target_attr = "RawScore"

We can now make a call to :code:`stat_distance` which will automatically choose the best
distance metric for us based on the distribution of the target attribute.

.. ipython:: python

  stat_distance(df, target_attr, group1, group2, mode="auto")

We can see that the distance between the groups is the same as above. :code:`stat_distance` has
chosen :code:`KolmogorovSmirnovDistance` as the best metric since the target column is continous.

It is possible to get a p-value back with the distance by using the :code:`p_value` flag.

.. ipython:: python

  stat_distance(df, target_attr, group1, group2, mode="auto", p_value=True)

Additional parameters are passed to the :code:`__init__` function of the distance metric. This can
be used to pass keyword arguments such as :code:`bin_edges` to categorical distance metrics.

.. ipython:: python
  :verbatim:

  _, bin_edges = np.histogram(df[target_attr], bins="auto")
  stat_distance(df, target_attr, group1, group2, mode="emd_categorical", bin_edges=bin_edges)

The distance metrics inside :code:`fairlens.bias.metrics` are also available for direct usage.

.. ipython:: python
  :verbatim:

  from fairlens.bias.metrics import (
      EarthMoversDistanceCategorical as EMD,
      KolmogorovSmirnovDistance as KS,
      LNorm
  )

  x = df[df["Ethnicity"] == "African-American"][target_attr]
  y = df[df["Ethnicity"] == "Caucasian"][target_attr]

  KS()(x, y)

  _, bin_edges = np.histogram(df[target_attr], bins="auto")
  EMD(bin_edges)(x, y)

  ord = 1
  LNorm(ord=ord)(x, y)
