from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*"
)

# =====================================================
# LOAD ML MODEL
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ml",
    "aerosense_aqi_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    print("AeroSense ML model loaded successfully.")
except Exception as e:
    print("ERROR LOADING ML MODEL:", e)
    raise


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
# HOME
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


# =====================================================
# ESP32 DATA UPLOAD
# =====================================================

@app.route("/api/upload", methods=["POST"])
def upload_data():

    try:

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
        # -------------------------------------------------

        pm25 = float(data.get("pm25", 0))
        pm10 = float(data.get("pm10", 0))
        humidity = float(data.get("rel_humidity", 0))
        temperature = float(data.get("temperature", 0))

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
        # ML PREDICTION
        # -------------------------------------------------

        predicted_aqi = model.predict(input_data)[0]
        predicted_aqi = round(float(predicted_aqi))

        # -------------------------------------------------
        # CATEGORY
        # -------------------------------------------------

        category = get_aqi_category(predicted_aqi)

        # -------------------------------------------------
        # ADVICE
        # -------------------------------------------------

        advice = get_advice(predicted_aqi)

        # -------------------------------------------------
        # ADD RESULTS
        # -------------------------------------------------

        data["ml_prediction"] = predicted_aqi
        data["ml_category"] = category
        data["advice"] = advice

        # -------------------------------------------------
        # SEND TO DASHBOARD
        # -------------------------------------------------

        socketio.emit("live_data", data)

        print("ML Predicted AQI:", predicted_aqi)
        print("Category:", category)

        return jsonify({
            "status": "success",
            "message": "Data received and ML prediction generated",
            "ml_prediction": predicted_aqi,
            "ml_category": category,
            "advice": advice
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# =====================================================
# LOCAL RUN
# =====================================================

if __name__ == "__main__":

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
