import pandas as pd
import sys

REQUIRED_COLUMNS = [
    'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Property_Area', 'Loan_Status'
]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw loan data. Raises AssertionError on structural problems,
    and drops/fixes rows that fail value-level checks."""

    assert not df.empty, "Dataframe is empty"

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Applicant income should never be negative
    df = df[df['ApplicantIncome'] >= 0]

    # Co-applicant income should never be negative
    df = df[df['CoapplicantIncome'] >= 0]

    # Loan amount, where present, should be positive
    df = df[(df['LoanAmount'].isna()) | (df['LoanAmount'] > 0)]

    # Credit history should only be 0, 1, or missing
    df = df[df['Credit_History'].isin([0.0, 1.0]) | df['Credit_History'].isna()]

    # Property_Area should be one of the known categories
    valid_areas = {'Urban', 'Rural', 'Semiurban'}
    assert set(df['Property_Area'].dropna().unique()).issubset(valid_areas), \
        "Unexpected category found in Property_Area"

    # Loan_Status should only be Y or N
    assert set(df['Loan_Status'].dropna().unique()).issubset({'Y', 'N'}), \
        "Unexpected value in Loan_Status"

    return df.reset_index(drop=True)


if __name__ == "__main__":
    input_path = "data/raw/loan_data.csv"
    output_path = "data/processed/validated.csv"

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    df_valid = validate(df)
    print(f"{len(df_valid)} rows passed validation "
          f"({len(df) - len(df_valid)} rows dropped)")

    df_valid.to_csv(output_path, index=False)
    print(f"Saved validated data to {output_path}")