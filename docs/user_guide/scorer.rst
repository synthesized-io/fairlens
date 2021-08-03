Fairness Scorer
===============

Users can use the :code:`FairnessScorer` to automatically analyze the dataset and look for inherent biases and hidden correlations.


Demographic Score
^^^^^^^^^^^^^^^^^

The :code:`distribution_score` method in the fairness scorer allows us to detect biases by measuring
whether the distribution of a target variable (score) relative to a demographic (Asian males), is
similar to the rest of the distribution. This is done by choosing a suitable distance metric based
on the target variable, and testing the significance of the distance between the distributions of
the demographic and the rest of the population. In the below case, our target variable, score,
is continuous, so the Kolmogrov Smirnov test is carried out. This process is done on all possible
demographics for the given sensitive attributes.

Let's test this out the on the compas dataset.

.. ipython:: python

  import pandas as pd
  from fairlens.scorer.fairness_scorer import FairnessScorer

  df = pd.read_csv("../datasets/compas.csv")
  df

  sensitive_attrs = ["Ethnicity", "Sex"]
  target_attr = "RawScore"

  fscorer = FairnessScorer(df, target_attr, sensitive_attrs)
  fscorer.distribution_score()

Generate Report
^^^^^^^^^^^^^^^

The object :code:`fairlens.FairnessScorer` can be used to automatically generate a fairness report on a
dataset with provided a target attribute.

.. ipython:: python

  import pandas as pd
  import fairlens as fl

  df = pd.read_csv("../datasets/german_credit_data.csv")
  df.info()

  @savefig quickstart_demographic_report.png
  fscorer = fl.FairnessScorer(df, "Credit amount").demographic_report()
