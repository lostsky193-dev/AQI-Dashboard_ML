import joblib
import pandas as pd

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model_path = "ml/aerosense_aqi_model.pkl"

model = joblib.load(model_path)


# ==========================================
# NEW SENSOR READINGS
# ==========================================

pm25 = 76.4
pm10 = 112.0
humidity = 68.0
temperature = 31.2


# ==========================================
# PREPARE INPUT
# ==========================================

input_data = pd.DataFrame([{
    "pm25": pm25,
    "pm10": pm10,
    "rel_humidity": humidity,
    "temperature": temperature
}])


# ==========================================
# PREDICT AQI
# ==========================================

predicted_aqi = model.predict(input_data)[0]

predicted_aqi = round(predicted_aqi)


# ==========================================
# AQI CATEGORY
# ==========================================

if predicted_aqi <= 50:
    category = "Good"
elif predicted_aqi <= 100:
    category = "Satisfactory"
elif predicted_aqi <= 200:
    category = "Moderate"
elif predicted_aqi <= 300:
    category = "Poor"
elif predicted_aqi <= 400:
    category = "Very Poor"
else:
    category = "Severe"


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n======================================")
print("       AEROSENSE ML PREDICTION")
print("======================================")

print(f"PM2.5        : {pm25}")
print(f"PM10         : {pm10}")
print(f"Humidity     : {humidity}%")
print(f"Temperature  : {temperature}°C")

print("\nPredicted AQI:", predicted_aqi)
print("Category:", category)