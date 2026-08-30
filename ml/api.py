from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ==========================================
# CREATE FASTAPI APP
# ==========================================

app = FastAPI(
    title="AeroSense AQI Prediction API",
    description="ML-based AQI prediction service",
    version="1.0"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "ml/aerosense_aqi_model.pkl"
)


# ==========================================
# INPUT DATA FORMAT
# ==========================================

class SensorData(BaseModel):
    pm25: float
    pm10: float
    rel_humidity: float
    temperature: float


# ==========================================
# AQI CATEGORY
# ==========================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AeroSense AQI Prediction API is running"
    }


# ==========================================
# AQI PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(data: SensorData):

    input_data = pd.DataFrame([{
        "pm25": data.pm25,
        "pm10": data.pm10,
        "rel_humidity": data.rel_humidity,
        "temperature": data.temperature
    }])

    prediction = model.predict(input_data)[0]

    predicted_aqi = round(prediction)

    category = get_aqi_category(predicted_aqi)

    return {
        "predicted_aqi": predicted_aqi,
        "category": category,
        "input": {
            "pm25": data.pm25,
            "pm10": data.pm10,
            "rel_humidity": data.rel_humidity,
            "temperature": data.temperature
        }
    }