Quickstart
==========

Installation
------------

FairLens can be installed using pip

.. code-block:: bash

  pip install fairlens


Generate Report
---------------

The object :code:`fairlens.FairnessScorer` can be used to automatically generate a fairness report on a
dataset, provided a target attribute.

.. ipython:: python

  import pandas as pd
  import fairlens as fl

  df = pd.read_csv("../datasets/german_credit_data.csv")
  df.info()

  @savefig quickstart_demographic_report.png
  fscorer = fl.FairnessScorer(df, "Credit amount").demographic_report()
