# app/routers/predict.py
from fastapi import APIRouter
from app.schemas import TextInput, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input: TextInput):
    return service.predict(input.text)
