import pandas as pd
import pytest
from src.validation.validate import validate


def make_base_df(**overrides):
    base = {
        'Loan_ID': ['LP001'],
        'Gender': ['Male'],
        'Married': ['Yes'],
        'Dependents': ['0'],
        'Education': ['Graduate'],
        'Self_Employed': ['No'],
        'ApplicantIncome': [5000],
        'CoapplicantIncome': [0],
        'LoanAmount': [120.0],
        'Loan_Amount_Term': [360.0],
        'Credit_History': [1.0],
        'Property_Area': ['Urban'],
        'Loan_Status': ['Y'],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_passes_clean_row():
    df = make_base_df()
    result = validate(df)
    assert len(result) == 1


def test_validate_missing_column_raises():
    df = make_base_df()
    df = df.drop(columns=['Credit_History'])
    with pytest.raises(AssertionError):
        validate(df)


def test_validate_drops_negative_applicant_income():
    df = make_base_df(ApplicantIncome=[-500])
    result = validate(df)
    assert len(result) == 0


def test_validate_drops_negative_coapplicant_income():
    df = make_base_df(CoapplicantIncome=[-200])
    result = validate(df)
    assert len(result) == 0


def test_validate_drops_zero_loan_amount():
    df = make_base_df(LoanAmount=[0])
    result = validate(df)
    assert len(result) == 0


def test_validate_allows_missing_loan_amount():
    df = make_base_df(LoanAmount=[None])
    result = validate(df)
    assert len(result) == 1


def test_validate_removes_duplicates():
    df = pd.concat([make_base_df(), make_base_df()], ignore_index=True)
    result = validate(df)
    assert len(result) == 1


def test_validate_rejects_invalid_property_area():
    df = make_base_df(Property_Area=['Moon'])
    with pytest.raises(AssertionError):
        validate(df)


def test_validate_rejects_invalid_loan_status():
    df = make_base_df(Loan_Status=['Maybe'])
    with pytest.raises(AssertionError):
        validate(df)