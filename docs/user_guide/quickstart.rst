Quickstart
==========

Installation
------------

FairLens can be installed using pip

.. code-block:: bash

  pip install fairlens


Assessing Fairness
------------------

The class :code:`fairlens.FairnessScorer` can be used to automatically generate a fairness report on a
dataset, provided a target column. The target column is a column we want to be independent of
of the sensitive attributes. We can analyze inherent biases in a dataset used for supervised learning
by passing in the the name of a desired output column. Alternatively, to assess the fairness of a
machine learning task on a dataset we can pass in the predicted column instead.

Below we show an assessment of fairness on a subset of Propublica's COMPAS dataset with respect to the attribute
"RawScore", which was indicative of an offender's likelihood to reoffend.

.. ipython:: python

  import pandas as pd
  import fairlens as fl

  df = pd.read_csv("../datasets/compas.csv")
  df.info()

  fscorer = fl.FairnessScorer(df, "RawScore")

  fscorer.sensitive_attrs

  @savefig quickstart_plot_distributions.png
  fscorer.plot_distributions()

From the plots, we can start to get an idea of where certain prejudices may occur. However, what's more
important is that we aren't prejudicing a very specific subgroup of sensitive attributes.
The fairness scorer can iterate through all possible combinations of these sensitive demographics
and use statistical distances to produce an estimate of how independent
the distribution of "RawScore" is to the sensitive groups.

At this point we may also find that we want to focus on a subset of the sensitive attributes, in which case we
can reinitialize the fairness scorer with new ones.

.. ipython:: python

  fscorer = fl.FairnessScorer(df, "RawScore", ["Ethnicity", "Sex", "MaritalStatus"])

  fscorer.demographic_report(max_rows=20)

This gives us a much clearer picture and both of the above suggest that there is a tendency
for "RawScore" being higher for people of in the group "African-American" and "Male"
than in the rest of the data.

.. note::

  As you may be able to tell from above, for many distance metrics there isn't a concept
  of positive or negative distances, which means that the fairness scorer will flag a skew
  in either direction.
