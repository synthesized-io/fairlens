Fairness and Bias
=================

This is a general guide to assessing fairness and bias in supervised learning tasks (classification, regression)
using structural datasets.

Literature Review
-----------------

In most supervised learning problems, a model is trained on a set of features :math:`X`, to predict or estimate
a target variable :math:`Y`. The resulting prediction of a trained model is denoted by :math:`R`. Additionally, we
define :math:`A` as a subset of :math:`X`, which corresponds to legally protected attributes such as
ethnicity, gender, etc.

.. math::

  \underbrace{\text{Title}\hspace{2mm}\overbrace{\text{Gender}\hspace{2mm} \text{Ethnicity}}^{A}\hspace{2mm}\text{Legal Status}}_{X}\hspace{3mm}\overbrace{\text{Raw Score}}^{Y}\hspace{3mm}\overbrace{\text{Predicted Score}}^{R}

There are multiple definitions of fairness in literature

There is a large amount of literature on fairness and bias, in the context of data science and
machine learning, in academia.

There is a large number of research papers covering bias and fairness in AI and methods of
reducing the problems of their presence in datasets or machine learning models.
Some of them involve data preprocessing, however those solutions might sometimes be problematic
due to hidden correlations present between sensitive groups and columns that seem inoffensive.

Algorithmic Fairness
^^^^^^^^^^^^^^^^^^^^

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

- Direct prejudice occurs when a sensitive column is directly used in a prediction model therefore the
  classification results will depend on sensitive features producing direct discrimination.
- Indirect prejudice is defined as the presence of a statistical dependence between a sensitive feature and
  the target. This results in a high correlation between the values of the sensitive column and the target.
  Even if the target is no longer a function of the sensitive data, it is still clearly influenced by it.
- Latent prejudice is defined as a statistical dependence between a sensitive variable and a non-sensitive one.
  For example, if there exists a correlation between a sensitive column and a non-sensitive column
  which is then used by the predictor, we can say the determination is discriminatory, as the
  dependence is still present, but a layer deeper.

Underestimation
^^^^^^^^^^^^^^^

Underestimation appears when a model is not fully converged due to the size limitations of a training dataset. Assuming
the presence of a an algorithm that is able to learn without commiting indirect prejudice, it is going to produce a fair
predictor if the the training examples are infinite.

However, if the size of the data is limited and some groups are underrepresented, the model might unintentionally produce
unfair results based on the local observations of the sample distributions.

Negative Legacy
^^^^^^^^^^^^^^^

Negative legacy is generated by biased sampling or labelling in a dataset. This can appear in data collected by organizations
that, historically, have been discriminatory towards minorities, without assessing individuals based on objective criteria. This
results in less samples of positive outcomes for some groups, leading to sample selection bias which is considerably difficult to
detect in training data.

References
----------

[1] Mehrabi N, Morstatter F, Saxena N, Lerman K, Galstyan A. A survey on bias and fairness in machine learning.
ACM Computing Surveys (CSUR). 2021 Jul 13;54(6):1-35.


[1] Kamishima, T., Akaho, S., and Sakuma, J. Fairness aware learning through regularization approach.
In IEEE 11th International Conference on Data Mining, pp. 643–650, 2011.

[2] Garg, Pratyush, John Villasenor, and Virginia Foggo. Fairness metrics: A comparative analysis.
In 2020 IEEE International Conference on Big Data (Big Data), pp. 3662-3666. IEEE, 2020.
