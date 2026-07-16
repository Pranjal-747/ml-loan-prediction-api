from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Loan Default Predictor API")

# Load model and column list ONCE at startup (not per-request — that would be slow)
model = joblib.load("loan_model.pkl")
model_columns = joblib.load("model_columns.pkl")


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def health_check():
    return {"status": "Loan Default Predictor API is live"}


@app.post("/predict")
def predict_loan(application: LoanApplication):
    # Convert incoming JSON into a DataFrame
    input_df = pd.DataFrame([application.dict()])

    # One-hot encode the same way training data was encoded
    input_df = pd.get_dummies(input_df)

    # Align columns: add any missing one-hot columns as 0, drop extras, keep exact training order
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict class and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of class "1" (default/approved)

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 4)
    }