Significance Testing
====================

Usually while measuring statistical distances and correlations users may want to test the significance
of such metrics.


Numerically Computing p-value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Say we want to test the hypothesis that the distribution of a variable in a subgroup is different to that
in the remaining population. For arbitrary metrics and statistics, we can use the data to tell us about
their distribution. This is done by using methods such as bootstrapping or permutation tests to resample
the data multiple times and recompute the statistic on each resampling, thereby providing an estimate for
its distribution. The distribution can then be used to compute a p-value or a confidence interval for the
metric.

Users can resample their data using methods from :code:`fairlens.bias.p-value` to provide an estimate for
the distribution of their test statistic. For instance if we want to obtain an estimate for the distribution
of the distance between the means of two subgroups of data using bootstrapping we can do the following.

.. ipython:: python

  import pandas as pd
  from fairlens.bias.p_value import bootstrap_statistic

  df = pd.read_csv("../datasets/compas.csv")

  group1 = df[df["Sex"] == "Male"]["RawScore"]
  group2 = df[df["Sex"] == "Female"]["RawScore"]
  test_statistic = lambda x, y: x.mean() - y.mean()

  t_distribution = bootstrap_statistic(group1, group2, test_statistic, n_samples=100)

  t_distribution

We can then compute a p-value by by inspecting the distribution of the test statistic.

.. ipython:: python

  from fairlens.bias.p_value import resampling_pvalue

  t_observed = test_statistic(group1, group2)

  resampling_pvalue(t_observed, t_distribution, alternative="two-sided")
