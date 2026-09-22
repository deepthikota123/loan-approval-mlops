import pandas as pd
from src.transformation.transform import fill_missing, engineer_features, encode_categoricals, transform


def make_base_df(**overrides):
    base = {
        'Loan_ID': ['LP001'],
        'Gender': ['Male'],
        'Married': ['Yes'],
        'Dependents': ['0'],
        'Education': ['Graduate'],
        'Self_Employed': ['No'],
        'ApplicantIncome': [5000],
        'CoapplicantIncome': [2000],
        'LoanAmount': [120.0],
        'Loan_Amount_Term': [360.0],
        'Credit_History': [1.0],
        'Property_Area': ['Urban'],
        'Loan_Status': ['Y'],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_fill_missing_fills_loan_amount():
    df = make_base_df(LoanAmount=[None])
    result = fill_missing(df)
    assert result['LoanAmount'].isna().sum() == 0


def test_fill_missing_fills_credit_history():
    df = make_base_df(Credit_History=[None])
    result = fill_missing(df)
    assert result['Credit_History'].isna().sum() == 0


def test_engineer_features_creates_total_income():
    df = make_base_df()
    df = fill_missing(df)
    result = engineer_features(df)
    assert 'TotalIncome_log' in result.columns
    assert 'LoanAmount_log' in result.columns
    # Raw columns should be dropped
    assert 'ApplicantIncome' not in result.columns
    assert 'CoapplicantIncome' not in result.columns
    assert 'Loan_ID' not in result.columns


def test_encode_categoricals_produces_numeric():
    df = make_base_df()
    df = fill_missing(df)
    df = engineer_features(df)
    encoded, encoders = encode_categoricals(df, fit=True)
    assert pd.api.types.is_numeric_dtype(encoded['Gender'])
    assert pd.api.types.is_numeric_dtype(encoded['Property_Area'])
    assert 'Gender' in encoders


def test_encode_categoricals_handles_three_plus_dependents():
    df = make_base_df(Dependents=['3+'])
    df = fill_missing(df)
    df = engineer_features(df)
    encoded, encoders = encode_categoricals(df, fit=True)
    # Should not raise, and Dependents should be numeric now
    assert pd.api.types.is_numeric_dtype(encoded['Dependents'])


def test_transform_end_to_end_produces_binary_target():
    df = make_base_df()
    result, encoders = transform(df, fit=True)
    assert set(result['Loan_Status'].unique()).issubset({0, 1})
    assert result.isna().sum().sum() == 0


def test_transform_reuses_encoders_for_new_data():
    df_train = pd.concat([make_base_df(), make_base_df(Property_Area=['Rural'])], ignore_index=True)
    _, encoders = transform(df_train, fit=True)

    df_new = make_base_df()
    df_new = df_new.drop(columns=['Loan_Status'])
    result, _ = transform(df_new, encoders=encoders, fit=False)
    assert pd.api.types.is_numeric_dtype(result['Property_Area'])