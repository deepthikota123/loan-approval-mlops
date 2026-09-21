import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = ['Gender', 'Married', 'Dependents', 'Education',
                     'Self_Employed', 'Property_Area']
TARGET_COL = 'Loan_Status'


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Categorical: fill with mode
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Numeric: fill with median
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Combine income into one feature
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Log-transform skewed numeric features (add 1 to avoid log(0))
    df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
    df['TotalIncome_log'] = np.log1p(df['TotalIncome'])

    # Drop raw skewed columns and ID (not predictive)
    df = df.drop(columns=['ApplicantIncome', 'CoapplicantIncome',
                           'LoanAmount', 'TotalIncome', 'Loan_ID'])

    return df


def encode_categoricals(df: pd.DataFrame, encoders=None, fit=True):
    df = df.copy()
    if encoders is None:
        encoders = {}

    # Dependents has "3+" as a category -> normalize to numeric-ish string first
    df['Dependents'] = df['Dependents'].replace('3+', '3')

    cols_to_encode = CATEGORICAL_COLS + ['Dependents']
    cols_to_encode = list(dict.fromkeys(cols_to_encode))  # dedupe, keep order

    for col in cols_to_encode:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


def transform(df: pd.DataFrame, encoders=None, fit=True):
    df = fill_missing(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)

    if TARGET_COL in df.columns and fit:
        df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})

    return df, encoders


if __name__ == "__main__":
    input_path = "data/processed/validated.csv"
    output_path = "data/processed/features.csv"

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    df_transformed, encoders = transform(df, fit=True)

    df_transformed.to_csv(output_path, index=False)
    joblib.dump(encoders, "models/encoders.pkl")

    print(f"Transformed data saved to {output_path}")
    print(f"Encoders saved to models/encoders.pkl")
    print(f"Final shape: {df_transformed.shape}")
    print(df_transformed.head())