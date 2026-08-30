import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

import joblib
import pandas as pd
import os
import json
import urllib.request
import urllib.parse


# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet"
)


# =====================================================
# LOAD ML MODEL
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ml",
    "aerosense_aqi_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("AeroSense ML model loaded successfully.")


# =====================================================
# AQI CATEGORY
# =====================================================

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


# =====================================================
# AI ADVICE
# =====================================================

def get_advice(aqi):

    if aqi <= 50:

        return [
            "Air quality is good.",
            "Normal outdoor activities are safe.",
            "No special precautions are required."
        ]

    elif aqi <= 100:

        return [
            "Air quality is satisfactory.",
            "Most people can continue normal outdoor activities.",
            "Sensitive individuals should monitor their exposure."
        ]

    elif aqi <= 200:

        return [
            "Reduce prolonged outdoor exposure.",
            "Sensitive individuals should take extra care.",
            "Consider limiting strenuous outdoor activity."
        ]

    elif aqi <= 300:

        return [
            "Avoid prolonged outdoor activity.",
            "Sensitive individuals should remain indoors when possible.",
            "Use a suitable mask during necessary outdoor travel."
        ]

    elif aqi <= 400:

        return [
            "Avoid outdoor activity as much as possible.",
            "Keep windows and doors closed during high-pollution periods.",
            "Sensitive individuals should remain indoors."
        ]

    else:

        return [
            "Avoid outdoor activity.",
            "Remain indoors and keep exposure to polluted air minimal.",
            "Follow local health advisories."
        ]


# =====================================================
# OPEN-METEO FUTURE AIR QUALITY FORECAST
# =====================================================

def get_location_forecast(latitude, longitude):

    try:

        # -------------------------------------------------
        # Validate coordinates
        # -------------------------------------------------

        latitude = float(latitude)
        longitude = float(longitude)

        if not (-90 <= latitude <= 90):
            return None

        if not (-180 <= longitude <= 180):
            return None

        # -------------------------------------------------
        # Open-Meteo Air Quality API
        #
        # Forecast is location based using GPS coordinates.
        #
        # We request 7 hourly points so we can show:
        # +1 hour
        # +3 hours
        # +6 hours
        # -------------------------------------------------

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "us_aqi,pm2_5,pm10,ozone",
            "forecast_hours": 7,
            "timezone": "auto"
        }

        url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality?"
            + urllib.parse.urlencode(params)
        )

        print("Requesting location forecast:")
        print(url)

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AeroSense-AQI/1.0"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=10
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        # -------------------------------------------------
        # Check response
        # -------------------------------------------------

        hourly = result.get("hourly", {})

        times = hourly.get("time", [])
        aqi_values = hourly.get("us_aqi", [])
        pm25_values = hourly.get("pm2_5", [])
        pm10_values = hourly.get("pm10", [])
        ozone_values = hourly.get("ozone", [])

        if len(aqi_values) == 0:
            print("No forecast AQI returned.")
            return None

        # -------------------------------------------------
        # Build +1h / +3h / +6h forecast
        # -------------------------------------------------

        forecast = []

        requested_hours = [1, 3, 6]

        for hour in requested_hours:

            if hour >= len(aqi_values):
                continue

            future_aqi = aqi_values[hour]

            forecast.append({
                "hours_ahead": hour,
                "time": (
                    times[hour]
                    if hour < len(times)
                    else None
                ),
                "aqi": (
                    round(float(future_aqi))
                    if future_aqi is not None
                    else None
                ),
                "category": (
                    get_aqi_category(round(float(future_aqi)))
                    if future_aqi is not None
                    else "Unavailable"
                ),
                "pm25": (
                    round(float(pm25_values[hour]), 1)
                    if hour < len(pm25_values)
                    and pm25_values[hour] is not None
                    else None
                ),
                "pm10": (
                    round(float(pm10_values[hour]), 1)
                    if hour < len(pm10_values)
                    and pm10_values[hour] is not None
                    else None
                ),
                "ozone": (
                    round(float(ozone_values[hour]), 1)
                    if hour < len(ozone_values)
                    and ozone_values[hour] is not None
                    else None
                )
            })

        return {
            "source": "Open-Meteo",
            "latitude": latitude,
            "longitude": longitude,
            "forecast": forecast
        }

    except Exception as e:

        print("Forecast error:", e)

        return None


# =====================================================
# HOME
# =====================================================

@app.route("/")
def index():

    return render_template("index.html")


# =====================================================
# OPTIONAL FORECAST ENDPOINT
# =====================================================

@app.route("/api/forecast", methods=["GET"])
def forecast_endpoint():

    try:

        latitude = request.args.get("latitude")
        longitude = request.args.get("longitude")

        if latitude is None or longitude is None:

            return jsonify({
                "status": "error",
                "message": "latitude and longitude are required"
            }), 400

        forecast = get_location_forecast(
            latitude,
            longitude
        )

        if forecast is None:

            return jsonify({
                "status": "error",
                "message": "Unable to retrieve forecast"
            }), 502

        return jsonify({
            "status": "success",
            "forecast": forecast
        })

    except Exception as e:

        print("Forecast endpoint error:", e)

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# =====================================================
# ESP32 DATA UPLOAD
# =====================================================

@app.route("/api/upload", methods=["POST"])
def upload_data():

    try:

        # -------------------------------------------------
        # GET JSON DATA
        # -------------------------------------------------

        data = request.get_json(silent=True)

        if not data:

            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        print("\nReceived from ESP32:")
        print(data)

        # -------------------------------------------------
        # GET SENSOR VALUES
        #
        # ESP32 currently sends:
        # pm25
        # pm10
        # temp
        # hum
        # -------------------------------------------------

        pm25 = float(
            data.get("pm25", 0)
        )

        pm10 = float(
            data.get("pm10", 0)
        )

        humidity = float(
            data.get("hum", 0)
        )

        temperature = float(
            data.get("temp", 0)
        )

        # -------------------------------------------------
        # ML INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame([{
            "pm25": pm25,
            "pm10": pm10,
            "rel_humidity": humidity,
            "temperature": temperature
        }])

        # -------------------------------------------------
        # PRESENT AQI FROM YOUR ML MODEL
        # -------------------------------------------------

        predicted_aqi = model.predict(
            input_data
        )[0]

        predicted_aqi = round(
            float(predicted_aqi)
        )

        # -------------------------------------------------
        # PRESENT AQI CATEGORY
        # -------------------------------------------------

        category = get_aqi_category(
            predicted_aqi
        )

        # -------------------------------------------------
        # PRESENT AQI ADVICE
        # -------------------------------------------------

        advice = get_advice(
            predicted_aqi
        )

        # -------------------------------------------------
        # GPS DATA FROM ESP32
        # -------------------------------------------------

        gps_fix = data.get(
            "gps_fix",
            False
        )

        latitude = data.get(
            "latitude"
        )

        longitude = data.get(
            "longitude"
        )

        # -------------------------------------------------
        # FUTURE LOCATION FORECAST
        # -------------------------------------------------

        location_forecast = None

        if (
            gps_fix is True
            and latitude is not None
            and longitude is not None
        ):

            location_forecast = get_location_forecast(
                latitude,
                longitude
            )

        # -------------------------------------------------
        # ADD ML RESULTS
        # -------------------------------------------------

        data["ml_prediction"] = predicted_aqi
        data["ml_category"] = category
        data["advice"] = advice

        # -------------------------------------------------
        # DASHBOARD FIELDS
        # -------------------------------------------------

        data["final_aqi"] = predicted_aqi
        data["category"] = category

        data["temp"] = temperature
        data["hum"] = humidity

        data["dominant"] = "ML AQI"

        # -------------------------------------------------
        # FUTURE FORECAST
        # -------------------------------------------------

        if location_forecast is not None:

            data["future_forecast"] = (
                location_forecast["forecast"]
            )

            data["forecast_source"] = (
                location_forecast["source"]
            )

        else:

            data["future_forecast"] = []
            data["forecast_source"] = None

        # -------------------------------------------------
        # BROADCAST TO DASHBOARD
        # -------------------------------------------------

        socketio.emit(
            "live_data",
            data
        )

        # -------------------------------------------------
        # PRINT
        # -------------------------------------------------

        print(
            "Present ML AQI:",
            predicted_aqi
        )

        print(
            "Present category:",
            category
        )

        print(
            "Advice:",
            advice
        )

        print(
            "GPS fix:",
            gps_fix
        )

        if latitude is not None:
            print(
                "Latitude:",
                latitude
            )

        if longitude is not None:
            print(
                "Longitude:",
                longitude
            )

        if location_forecast is not None:

            print(
                "Future forecast:",
                location_forecast["forecast"]
            )

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status":
                "success",

            "message":
                "Data received and AQI analysis generated",

            # Present AQI
            "present_aqi":
                predicted_aqi,

            "present_category":
                category,

            # Keep existing names
            "ml_prediction":
                predicted_aqi,

            "ml_category":
                category,

            "final_aqi":
                predicted_aqi,

            "category":
                category,

            "advice":
                advice,

            # GPS
            "gps_fix":
                gps_fix,

            "latitude":
                latitude,

            "longitude":
                longitude,

            # Future
            "future_forecast":
                (
                    location_forecast["forecast"]
                    if location_forecast is not None
                    else []
                ),

            "forecast_source":
                (
                    location_forecast["source"]
                    if location_forecast is not None
                    else None
                ),

            "data":
                data
        })

    except Exception as e:

        print(
            "UPLOAD ERROR:",
            e
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)
        }), 400


# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":

    socketio.run(

        app,

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False
    )
