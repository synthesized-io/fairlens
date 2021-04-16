import pandas as pd
import pytest

from fairlens.transformer import FairnessTransformer


@pytest.mark.parametrize(
    "file_name,sensitive_attributes,target,target_n_bins",
    [
        pytest.param("data/credit_with_categoricals.csv", ["age"], "SeriousDlqin2yrs", None, id="binary_target"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "MonthlyIncome", None, id="continuous_target"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "MonthlyIncome", 5, id="continuous_target_binned"),
        pytest.param("data/credit_with_categoricals.csv", ["age"], "effort", None, id="multinomial_target"),
        pytest.param(
            "data/claim_prediction.csv",
            ["age", "sex", "children", "region"],
            "insuranceclaim",
            None,
            id="claim_prediction",
        ),
        pytest.param("data/claim_prediction.csv", [], "insuranceclaim", None, id="no_sensitive_attrs"),
        pytest.param("data/biased_data_mixed_types.csv", ["age", "gender", "DOB"], "income", None, id="mixed_types"),
        pytest.param(
            "data/biased_data_mixed_types.csv", ["age", "gender", "DOB"], "income", 5, id="mixed_types_target_binned"
        ),
    ],
)
def test_fairness_transformer(file_name, sensitive_attributes, target, target_n_bins):
    df = pd.read_csv(file_name)
    sample_size = 10_000
    df = df.sample(sample_size) if len(df) > sample_size else df

    ft = FairnessTransformer(sensitive_attrs=sensitive_attributes, target=target, target_n_bins=target_n_bins)
    ft.fit(df)
    df_t = ft.transform(df)

    min_num_unique: int = 10
    categorical_threshold_log_multiplier: float = 2.5

    categorical_threshold = int(max(min_num_unique, categorical_threshold_log_multiplier))

    for col in sensitive_attributes:
        assert df_t[col].dtype.kind == "O"
        if df[col].nunique() > categorical_threshold:
            assert df_t.loc[df_t[col] != "nan", col].nunique() == 5

    assert df_t[target].dtype.kind == "O"
    if target_n_bins:
        assert df_t[target].nunique() <= target_n_bins
