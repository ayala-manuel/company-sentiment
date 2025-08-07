# app/schemas.py

from pydantic import BaseModel
from typing import List

class TextInput(BaseModel):
    text: str

class SentenceSentiment(BaseModel):
    sentence: str
    label: str
    score: int

class PredictionResponse(BaseModel):
    overall_sentiment: dict
    sentence_sentiments: List[SentenceSentiment]