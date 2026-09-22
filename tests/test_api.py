from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_endpoint_returns_valid_response():
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ["Approved", "Rejected"]
    assert 0 <= body["probability_approved"] <= 1
    assert 0 <= body["probability_rejected"] <= 1


def test_predict_endpoint_rejects_missing_field():
    payload = {
        "Gender": "Male",
        "Married": "Yes",
        # Dependents missing
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_endpoint_low_credit_history_influences_result():
    payload_good_credit = {
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 5000, "CoapplicantIncome": 0,
        "LoanAmount": 120, "Loan_Amount_Term": 360,
        "Credit_History": 1.0, "Property_Area": "Urban",
    }
    payload_bad_credit = {**payload_good_credit, "Credit_History": 0.0}

    resp_good = client.post("/predict", json=payload_good_credit).json()
    resp_bad = client.post("/predict", json=payload_bad_credit).json()

    # Good credit history should generally yield higher approval probability
    assert resp_good["probability_approved"] >= resp_bad["probability_approved"]


def test_metrics_endpoint_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200