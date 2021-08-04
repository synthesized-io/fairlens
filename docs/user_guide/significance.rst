Significance Tests
==================

Usually while measuring statistical distances and correlations users may want to test the significance
of such metrics.

Resampling
^^^^^^^^^^

Say we want to test the hypothesis that the distribution of a variable in a subgroup is different to that
in the remaining population. For arbitrary metrics and statistics, we can use the data to tell us about
their distribution. This is done by using methods such as bootstrapping or permutation tests to resample
the data multiple times and recompute the statistic on each sample, thereby providing an estimate for
its distribution. The distribution can then be used to compute a p-value or a confidence interval for the
metric.

Users can resample their data using methods from :code:`fairlens.bias.p-value` to provide an estimate for
the distribution of their test statistic. For instance if we want to obtain an estimate for the distribution
of the distance between the means of two subgroups of data using bootstrapping or permutation testing
we can do the following.

.. ipython:: python
  import pandas as pd
  from fairlens.bias.p_value import bootstrap_statistic, permutation_statistic

  df = pd.read_csv("../datasets/compas.csv")

  group1 = df[df["Sex"] == "Male"]["RawScore"]
  group2 = df[df["Sex"] == "Female"]["RawScore"]
  test_statistic = lambda x, y: x.mean() - y.mean()

  t_distribution = permutation_statistic(group1, group2, test_statistic, n_perm=100)
  t_distribution

  t_distribution = bootstrap_statistic(group1, group2, test_statistic, n_samples=100)
  t_distribution


Intervals and P-Values
^^^^^^^^^^^^^^^^^^^^^^

The distribution of the test statistic produced by resampling can be used to compute a confidence
interval or a p-value. We can use our bootstrapped distribution from above to do so using the
following methods.

.. ipython:: python

  from fairlens.bias.p_value import resampling_interval

  t_observed = test_statistic(group1, group2)

  resampling_interval(t_observed, t_distribution, cl=0.95)

.. ipython:: python

  from fairlens.bias.p_value import resampling_p_value

  resampling_p_value(t_observed, t_distribution, alternative="two-sided")
