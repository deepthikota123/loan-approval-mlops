from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

from src.transformation.transform import transform

app = FastAPI(title="Loan Approval Prediction API", version="1.0.0")

# Load model + encoders once, at startup
model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")


class LoanApplication(BaseModel):
    Gender: str = Field(..., examples=["Male"])
    Married: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["0"])
    Education: str = Field(..., examples=["Graduate"])
    Self_Employed: str = Field(..., examples=["No"])
    ApplicantIncome: float = Field(..., examples=[5000])
    CoapplicantIncome: float = Field(..., examples=[0])
    LoanAmount: float = Field(..., examples=[120])
    Loan_Amount_Term: float = Field(..., examples=[360])
    Credit_History: float = Field(..., examples=[1.0])
    Property_Area: str = Field(..., examples=["Urban"])


@app.get("/")
def root():
    return {"message": "Loan Approval Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        data = application.dict()
        data["Loan_ID"] = "TEMP"  # placeholder, dropped during transform
        df = pd.DataFrame([data])

        transformed, _ = transform(df, encoders=encoders, fit=False)

        pred = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0].tolist()

        return {
            "prediction": "Approved" if pred == 1 else "Rejected",
            "probability_rejected": round(proba[0], 4),
            "probability_approved": round(proba[1], 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


Instrumentator().instrument(app).expose(app)