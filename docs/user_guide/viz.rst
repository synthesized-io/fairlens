Visualizing distributions
=========================

Fairlens supports tools to visualize the distribution of sensitive groups relative to one another.

First we will import the required packages and load the compas dataset.

.. note::
  FairLens ships with a custom style which can be used. You can be activate this using :code:`fairlens.plot.use_style`,
  or reset to the defaults using :code:`fairlens.plot.reset_style`. Note that using these styles may override your
  systems default parameters. This may prove useful if text in plots is misaligned or too large.

.. ipython:: python

  import pandas as pd
  import fairlens as fl

  fl.plot.use_style()

  df = pd.read_csv("../datasets/compas.csv")
  df.info()


Distribution of Groups
----------------------

Visualizing the distribution of a variable in 2 distinct sub-groups can help us understand if the
variable is skewed in the direction of either of the sub-groups. For instance in the COMPAS dataset,
this can help in showing whether the distribution of raw COMPAS scores is skewed in the favor of
Caucasian Males as compared to African-American Males.
The method :code:`fairlens.plot.distr_plot` can be used to visualize the distributions of these groups.

.. ipython:: python

  import matplotlib.pyplot as plt

  target_attr = "RawScore"
  group1 = {"Sex": ["Male"], "Ethnicity": ["Caucasian"]}
  group2 = {"Sex": ["Male"], "Ethnicity": ["African-American"]}

  @savefig distr_plot.png
  fl.plot.distr_plot(df, target_attr, [group1, group2])

By default, this method plots a Kernel Density Estimator over the raw data, however it can be configured
to plot histograms instead. See the API-Reference for more details.

Distribution of Groups in a Column
----------------------------------

It can be insightful to visualize the distribution of a variable with respect to all
the unique sub-groups of data in a column.
For instance, visualizing the distribution of raw scores in the COMPAS dataset with respect
to all Ethnicities may help us understand the relationship between Ethnicity and raw scores.
The method :code:`fairlens.plot.attr_distr_plot` can be used for this.

.. ipython:: python

  target_attr = "RawScore"
  sensitive_attr = "Ethnicity"

  @savefig attr_distr_plot.png
  fl.plot.attr_distr_plot(df, target_attr, sensitive_attr)

Additionally, this can be extended to plot the distribution of raw scores with respect to
a list of sensitive attributes using :code:`fairlens.plot.mult_distr_plot`. This can be
used give a rough overview of the relationship between all sensitive attributes and a
target variable.

.. ipython:: python

  target_attr = "RawScore"
  sensitive_attrs = ["Ethnicity", "Sex"]

  @savefig mult_distr_plot.png
  fl.plot.mult_distr_plot(df, target_attr, sensitive_attrs)
