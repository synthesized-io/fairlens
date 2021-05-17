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
  :verbatim:

  from fairlens.bias.metrics import stat_distance

We need to define which groups of data we want to measure the distance between.

.. ipython:: python
  :verbatim:

  group1 = {"Ethnicity": ["African-American"]}
  group2 = {"Ethnicity": ["Caucasian"]}

We can now make a call to :code:`stat_distance` which will automatically choose the best
distance metric for us based on the distribution of the target attribute.

.. ipython:: python
  :verbatim:

  stat_distance(df, target_attr, group1, group2, mode="auto")
