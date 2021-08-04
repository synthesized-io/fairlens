Statistical Distances
=====================

FairLens allows users to make use of a wide range of statistical distance metrics to measure the difference
between the distributions of a variable in two potentially sensitive sub-groups of data. These metrics
are available for individual use in the package :code:`fairlens.metrics`, or they can be called using
the :code:`stat_distance` wrapper function.

Let's import this method and load in the compas dataset.

.. ipython:: python

  import pandas as pd
  import fairlens as fl

  df = pd.read_csv("../datasets/compas.csv")
  df.info()


Distance metrics from :code:`fairlens.metrics` can be used by passing two columns of data
to their :code:`__call__` method, which will return the respective metric or None if it
cannot be computed. Different metrics take in different keyword arguments which can
be passed to their constructor.

.. ipython:: python

  target_attr = "RawScore"
  x = df[df["Ethnicity"] == "African-American"][target_attr]
  y = df[df["Ethnicity"] == "Caucasian"][target_attr]

  fl.metrics.KolmogorovSmirnovDistance()(x, y)

  _, bin_edges = np.histogram(df[target_attr], bins="auto")
  fl.metrics.EarthMoversDistance(bin_edges)(x, y)

  ord = 1
  fl.metrics.Norm(ord=ord)(x, y)

The method :code:`stat_distance` provides a simplified wrapper for distance metrics
and allows us to define which sub-groups of data we want to measure the distance between
using a simplified dictionary notation or a predicate itself.

.. ipython:: python

  group1 = {"Ethnicity": ["African-American"]}
  group2 = df["Ethnicity"] == "Caucasian"

We can now make a call to :code:`stat_distance`. The parameter mode is indicative of the
statistical distance metric, and corresponds to the :code:`id` function in the classes
for the distance metrics. If it is set to "auto", a suitable distance metric will
be chosen depending on the distribution of the target variable.

.. ipython:: python

  fl.metrics.stat_distance(df, target_attr, group1, group2, mode="auto")

We can see that the distance between the groups is the same as above. :code:`stat_distance` has
chosen :code:`KolmogorovSmirnovDistance` as the best metric since the target column is continous.

It is possible to get a p-value back with the distance by using the :code:`p_value` flag.

.. ipython:: python

  fl.metrics.stat_distance(df, target_attr, group1, group2, mode="auto", p_value=True)

Additional parameters are passed to the :code:`__init__` function of the distance metric. This can
be used to pass keyword arguments such as :code:`bin_edges` to categorical distance metrics.

.. ipython:: python

  _, bin_edges = np.histogram(df[target_attr], bins="auto")
  fl.metrics.stat_distance(df, target_attr, group1, group2, mode="emd", bin_edges=bin_edges)
