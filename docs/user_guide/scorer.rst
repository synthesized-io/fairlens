Fairness Scorer
===============

The class :code:`fairlens.FairnessScorer` can be used to automatically analyze a dataset and assess fairness.

The fairness scorer takes in 2 parameters, the dataset in a dataframe, and a target variable. The target variable
can be the output column of the dataset or the result of a prediction made by a model on the dataset. The sensitive
attributes are automatically detected, but users can be explicit and pass them in. Additionally, the user can
choose to be explicit about the the type of data in each column i.e. categorical, continuous, binary, date;
however, by default this is automatically inferred.

The fairness scorer aims to measure group fairness; i.e. if the chosen target column is independent of the
sensitive attributes in a row.
If the distribution of the target column in each subgroup of sensitive values isn't significantly different to the
overall distribution, then we assume independence. Hence to find the most prejudiced subgroups we can
use statistical distances and hypothesis tests to measure the significance of the skew in each subgroup.

Report Generation
------------------

The fairness scorer supports three different methods for report generation.

The :code:`plot_distribution` method produces plots of the distribution of the target variable in each subgroup
in a column, for each column. This is useful for understanding the different distributions of protected groups
and identifying any inherent biases present in a dataset.

.. ipython:: python

  import pandas as pd
  import fairlens as fl

  df = pd.read_csv("../datasets/compas.csv")
  df.info()

  fscorer = fl.FairnessScorer(df, "RawScore", ["Ethnicity", "Sex"])


The :code:`demographic_report` estimates the extent to which the distribution of the target column is independent
of the sensitive attributes. This is done by using a suitable distance metric to test the significance of the
distance between the distributions of each senstitive demographic and the overall population. In the below case,
our target variable, "RawScore", is continuous, so the Kolmogorov Smirnov test is carried out by default.
This process is done on all possible demographics for the given sensitive attributes. This report produces a
list of the most prejudiced groups (i.e. largest distance / p-value) by this criterion.

.. ipython:: python

  fscorer.demographic_report()

Users can also opt to measure the distance between the distributions in the subgroup and the data without the subgroup
as an alternative.

.. ipython:: python

  fscorer.demographic_report(method="dist_to_rest")


Scoring API
-----------

Individual functions of the demographic report can be called for further analysis.

.. ipython:: python

  sensitive_attrs = ["Ethnicity", "Sex"]
  target_attr = "RawScore"

  fscorer = fl.FairnessScorer(df, target_attr, sensitive_attrs)
  df_dist = fscorer.distribution_score()

  df_dist
