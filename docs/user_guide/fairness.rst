Overview
========

In most supervised learning problems, a model is trained on a set of features :math:`X`, to predict or estimate
a target variable :math:`Y`. The resulting prediction of a trained model is denoted by :math:`R`. Additionally, we
define :math:`A`, a subset of :math:`X`, which corresponds to legally protected attributes.

.. raw:: html

  <div style="width: 100%; margin: auto; text-align: center;">

:math:`\underbrace{\text{Title}\quad \overbrace{\text{Gender}\quad \text{Ethnicity}}^{A}\quad \text{Legal Status}}_{X}\quad \overbrace{\text{Raw Score}}^{Y}\quad \overbrace{\text{Predicted Score}}^{R}` :footcite:p:`compas`

.. raw:: html

  </div>

There are multiple definitions of fairness in literature, and while many can be contentious, results are often
considered fair if they are independent of legally protected characteristics such as age, gender, and ethnicity
:footcite:p:`fairmlbook`.

.. math::

  P(R \mid A) = P(R)

In practice, most legally protected attributes tend to be categorical, therefore for the sake of simplicity
we model protected or sensitive variables as discrete random variables, which gives us the following for independence.

.. math::

  P(R \mid A = a) = P(R)\quad \forall a \in A

.. math::
  \text{or}

.. math::

  P(R \mid A = a) = P(R \mid A = b)\quad \forall a,b \in A

While :footcite:t:`fairmlbook` proposes 2 alternative fairness or non-discrimination criteria, separation and sufficiency,
which are applicable to a range of problems, we have chosen to work with independence because of its generality
:footcite:p:`gouic2020projection`.
Working with this abstract definition of fairness, we quantify the bias of a variable :math:`T` in a group :math:`a`,
as the statistical distance between the the probability distributions of :math:`P(T \mid A = a)` and :math:`P(T)`.

.. image:: ../_static/distance.png

Using this definition, we can carry out hypothesis tests to measure the significance of an observed bias in a variable
in a sensitive subgroup. The advantage of using this method is that the target variable can be continuous, categorical
or binary, depending on the metric or test used.

FairLens consists of a range of metrics and tests for statistical similarity and correlation, along with Monte Carlo
methods for hypothesis testing, which can be used to assess fairness in the aformentioned way. The fairness of
a variable, with respect to sensitive attributes, is assessed by measuring the bias of the variable in each
possible sensitive subgroup and taking the weighted average.

In practice, :math:`R` can be substituted with any column in the data. For instance, if :math:`R` is the target column
:math:`Y` in a dataset, then using the above methods will yeild results indicating inherent biases present in the
dataset. This can be useful for mitigating bias in the data itself, which may nip the problem in the bud.
On the other hand, using the predictions as :math:`R` will indicate the algorithmic bias of the model.

References
----------

.. footbibliography::
