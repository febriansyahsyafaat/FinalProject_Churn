from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import ChurnPredictor
import uvicorn

app = FastAPI(title="Bank Churn Prediction API", version="1.0.0")

predictor = ChurnPredictor(
    model_path='models/best_model.pkl',
    scaler_path='models/scaler.pkl',
    feature_names_path='models/feature_names.pkl'
)

class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    Gender: int = Field(..., ge=0, le=1)
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    Geography_Germany: int = Field(..., ge=0, le=1)
    Geography_Spain: int = Field(..., ge=0, le=1)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    churn_status: str
    risk_level: str

@app.get("/")
def read_root():
    return {"message": "Bank Churn Prediction API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        input_dict = customer.dict()
        prediction, probability = predictor.predict(input_dict)
        churn_status = "CHURN" if prediction == 1 else "NOT CHURN"
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "churn_status": churn_status,
            "risk_level": risk_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
