Visualizing distributions
=========================

Fairlens supports tools to visualize the distribution of sensitive groups relative to one another.

First we will import the required packages, load the compas dataset, and define our groups.

.. ipython:: python

  import pandas as pd
  from fairlens.bias.viz import use_style

  use_style()

  df = pd.read_csv("../datasets/compas.csv")
  df.info()

  target_attr = "RawScore"
  group1 = {"Ethnicity": ["African-American"]}
  group2 = {"Ethnicity": ["Caucasian"]}


Now we can use :code:`distr_plot` to visualize the distributions of these groups.

.. ipython:: python

  import matplotlib.pyplot as plt
  from fairlens.bias.viz import distr_plot

  @savefig distr_plot.png
  distr_plot(df, target_attr, [group1, group2])

  @verbatim
  plt.show()


Additionally, we may want to visualize the distribution of all unique values for sensitive attributes
relative to one another. We can use :code:`attr_distr_plot` to do this.

.. ipython:: python

  from fairlens.bias.viz import attr_distr_plot

  sensitive_attr = "Ethnicity"

  @savefig attr_distr_plot.png
  attr_distr_plot(df, target_attr, sensitive_attr)

  @verbatim
  plt.show()
