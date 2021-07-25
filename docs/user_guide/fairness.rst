Overview
========

Fairlens is an open-source Python package for automatically discovering hidden biases and measuring
fairness in datasets. The package makes use of a wide range of customisable statistical distance and
correlation metrics that are used to quickly identify bias, produce meaningful diagrams and measure
fairness across a configurable range of sensitive and legally protected categories such as ethnicity,
nationality or gender.

Literature Review
-----------------

There is a large number of research papers covering bias and fairness in AI and methods of
reducing the problems of their presence in datasets or machine learning models.
Some of them involve data preprocessing, however those solutions might sometimes be problematic
due to hidden correlations present between sensitive groups and columns that seem inoffensive.

However, the great majority of solutions for ensuring fairness involve algorithmic fairness, which
presumes the existence of a predictor or some type of machine learning model. Even more, it is a
requirement for applying methods that address bias issues through the need of actual predictions,
which are needed to calculate various probabilities, such as true and false positive rates, positive and
negative predictive values and many more. These are then used to enforce metrics such as equalized odds,
statistical or predictive parity.

Equalized odds
^^^^^^^^^^^^^^

Equalized odds is satisfied by a predictor when it produces a true positive rate and a false positive rate
that are equal across the sensitive groups. This implies that the model has the same chance of correctly
classifying a true positive as positive and incorrectly classifying a true positive as negative.

.. math::
    P(\hat{y} \mid y = 0, G = male) = P(\hat{y} \mid y = 0, G = female)

Statistical parity
^^^^^^^^^^^^^^^^^^

Alternatively, statistical parity (also know as group fairness or demographic parity) is achieved when members
of the different sub-groups of protected categories (i.e. male and female for the protected category of gender)
are predicted to belong to the positive class at the same rate.

.. math::
    P(\hat{y} \mid G = male) = P(\hat{y} \mid G = female)

It is clear to see that while both of these solutions seem fair, they constrain the model in different ways and
it has actually been shown that it is not mathematically possible to enforce two or more methods on the same model [2],
which raises a dilemma on the true meaning of bias and what type of fairness is the most desirable.

While any of methods can be employed to produce workable results in practice, they are not exhaustive, as some use cases
might not be focused on producing an ML model as an end result. Data synthesis is a good example of a different purpose,
as the desired result is a new dataset that accurately reflects the intrinsic traits and column distributions of the original data,
with the added benefits of resolving privacy and anonimity issues, as well as possibly extending the dataset to better reflect
underrepresented parts of the data, with the goal of becoming more relevant in decision-making computations.

As such, the aim of Fairlens is to analyze datasets and produce fairness scores in the absence of a machine learning model,
picking up discrepancies between different sensitive demographics with respect to a target column, usually representing
a score or classification.

Types of Biases
---------------

Some of the predominant causes of unfairness resulting from structural datasets are highlighted in the
works of Kamishima et al. on a regularization approach to fairness in models [1]. In summary, the
following are ways in which biases can exist in structural data.

Prejudice
^^^^^^^^^

Prejudice occurs when the target column in a dataset, or any column used in a prediction model, is
dependant on a sensitive column. There are three types of prejudice.

.. * Direct prejudice occurs when a sensitive column is directly used in a prediction model thereforex

Underestimation
^^^^^^^^^^^^^^^

Negative Legacy
^^^^^^^^^^^^^^^


References
----------

[1] Kamishima, T., Akaho, S., and Sakuma, J. Fairness aware learning through regularization approach.
In IEEE 11th International Conference on Data Mining, pp. 643–650, 2011.

[2] Garg, Pratyush, John Villasenor, and Virginia Foggo. Fairness metrics: A comparative analysis.
In 2020 IEEE International Conference on Big Data (Big Data), pp. 3662-3666. IEEE, 2020.
