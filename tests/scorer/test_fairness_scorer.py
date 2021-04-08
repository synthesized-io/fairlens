import pandas as pd
import pytest

from fairness.scorer import FairnessScorer
from fairness.scorer.fairness_transformer import ModelType
from tests.utils import progress_bar_testing


@pytest.mark.slow
@pytest.mark.parametrize(
    "file_name,sensitive_attributes,target,mode",
    [
        pytest.param("data/credit_with_categoricals.csv", ["age"], "SeriousDlqin2yrs", None, id="binary_target"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "RevolvingUtilizationOfUnsecuredLines", None,
                     id="continuous_target"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "effort", "emd", id="multinomial_target_emd"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "effort", "ovr", id="multinomial_target_ovr"),
        pytest.param("data/claim_prediction.csv", ["age", "sex", "children", "region"], "insuranceclaim",
                     None, id="claim_prediction"),
        pytest.param("data/claim_prediction.csv", [], "insuranceclaim", None, id="no_sensitive_attrs"),
        pytest.param("data/biased_data_mixed_types.csv", ["age", "gender", "DOB"], "income", None, id="mixed_types"),
    ]
)
def test_fairness_scorer_parametrize(file_name, sensitive_attributes, target, mode):

    data = pd.read_csv(file_name)
    sample_size = 10_000
    data = data.sample(sample_size) if len(data) > sample_size else data

    fairness_scorer = FairnessScorer(data, sensitive_attrs=sensitive_attributes, target=target)

    # Distributions Score
    if mode is None:
        mode = 'emd'
    dist_score, dist_biases = fairness_scorer.distributions_score(data, mode=mode,
                                                                  progress_callback=progress_bar_testing)

    assert 0. <= dist_score <= 1.
    assert not any([dist_biases[c].isna().any() for c in dist_biases.columns])
    assert all([isinstance(v, list) for v in dist_biases['name'].values])
    assert all([isinstance(v, list) for v in dist_biases['value'].values])
    assert not any(['nan' in v for v in dist_biases['value'].values])  # Shouldn't have biases for NaN values.


@pytest.mark.slow
@pytest.mark.parametrize(
    "file_name,sensitive_attributes,target",
    [
        pytest.param("data/claim_prediction.csv", ["age", "sex", "children", "region"], "insuranceclaim",
                     id="claim_prediction"),
        pytest.param("data/credit_with_categoricals.csv", [], "age", id="target_in_sensitive_attrs"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "MonthlyIncome", id="target_in_sensitive_attrs"),
    ]
)
def test_fairness_scorer_detect_sensitive(file_name, sensitive_attributes, target):
    data = pd.read_csv(file_name)

    fairness_scorer = FairnessScorer.init_detect_sensitive(data, target=target)
    assert fairness_scorer.sensitive_attrs == sensitive_attributes

    dist_score, dist_biases = fairness_scorer.distributions_score(data, progress_callback=progress_bar_testing)

    assert 0. <= dist_score <= 1.

    assert not any([dist_biases[c].isna().any() for c in dist_biases.columns])


def test_detect_sensitive():
    attrs = ["i", "love", "sex", "in", "any", "location"]

    sensitive_attrs = FairnessScorer.detect_sensitive_attrs(attrs)
    assert sensitive_attrs == ["sex", "location"]
