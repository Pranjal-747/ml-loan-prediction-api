# Loan Status Predictor API

A machine learning API that predicts loan approval likelihood based on applicant
financial and demographic details. 

## 🔗 Live Demo
- API Docs (Swagger UI): https://ml-loan-prediction-api.onrender.com/docs

> Note: hosted on Render's free tier — the app sleeps after 15 minutes of
> inactivity. The first request after idle may take 30-50 seconds to respond.

## 🧠 Problem Statement
Dream Housing Finance wants to automate loan eligibility decisions based on
applicant details submitted through an online form (income, credit history,
employment status, etc.), rather than manual review.

## 🏗️ Architecture

Raw CSV (loan_data.csv)
      │
      ▼
Feature Engineering
 ├── Missing value imputation (median / most frequent)
 └── One-hot encoding (categorical → numeric)
      │
      ▼
Random Forest Classifier (scikit-learn)
      │
      ▼
Model + column schema saved via Joblib
      │
      ▼
FastAPI /predict endpoint (Pydantic validation)
      │
      ▼
Dockerized
      │
      ▼
Deployed on Render.com


## 🛠️ Tech Stack
- ML: scikit-learn, pandas, NumPy
- API: FastAPI, Pydantic, Uvicorn
- Deployment: Docker, Render.com
- Model:Random Forest Classifier

## 📊 Model Performance
- Test Accuracy:82.8%
- Dataset: Dream Housing Finance Loan Prediction dataset (Analytics Vidhya)
- 491 rows, 11 input features, binary target (Loan_Status)

## 📥 API Usage

**Endpoint:** `POST /predict`

**Sample Request:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5849,
  "CoapplicantIncome": 0,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Urban"
}
```

**Sample Response:**
```json
{
  "prediction": 1,
  "meaning": "Approved",
  "confidence": 0.87
}
```

## 💻 Run Locally

```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]

# Option 1: Run with Docker (recommended)
docker build -t loan-predictor .
docker run -p 8000:8000 loan-predictor

# Option 2: Run directly with Python
pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` to test the API.

## 📁 Project Structure
```
├── main.py                 # FastAPI app + /predict endpoint
├── train_model.py          # Model training script
├── loan_data.csv           # Training dataset
├── loan_model.pkl          # Saved trained model
├── model_columns.pkl       # Saved column schema for inference
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── README.md
```

## 🎯 What This Project Demonstrates
- End-to-end ML pipeline: data cleaning → feature engineering → model training → evaluation
- Production API design with input validation (Pydantic)
- Containerization for portable, reproducible deployment
- Live cloud deployment (Render.com)

