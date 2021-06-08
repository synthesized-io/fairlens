Visualizing distributions
=========================

Fairlens supports tools to visualize the distribution of sensitive groups relative to one another.

First we will import the required packages, load the compas dataset, and define our groups.

.. ipython:: python

  import pandas as pd

  df = pd.read_csv("../datasets/compas.csv")
  df.head()

  target_attr = "RawScore"
  group1 = {"Ethnicity": ["African-American"]}
  group2 = {"Ethnicity": ["Caucasian"]}


Now we can use :code:`plt_group_dist` to visualize the distributions of these groups.

.. ipython:: python

  from fairlens.bias.viz import plt_group_dist

  @savefig plt_group_dist.png
  plt_group_dist(df, target_attr, group1, group2, legend=True)

  @verbatim
  plt.show()


Additionally, we may want to visualize the distribution of all unique values for sensitive attributes
relative to one another. We can use :code:`plt_attr_dist` to do this.

.. ipython:: python

  from fairlens.bias.viz import plt_attr_dist

  sensitive_attr = "Ethnicity"

  @savefig plt_attr_dist.png
  plt_attr_dist(df, target_attr, sensitive_attr, legend=True)

  @verbatim
  plt.show()
