# app/main.py
from fastapi import FastAPI
from app.routers.predict import router as predict_router
from app.routers.health import router as health_router

app = FastAPI()
app.include_router(predict_router)
app.include_router(health_router)