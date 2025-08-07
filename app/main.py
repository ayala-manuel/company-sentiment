# app/main.py
from fastapi import FastAPI
from app.routers.predict import router as predict_router

app = FastAPI()
app.include_router(predict_router)