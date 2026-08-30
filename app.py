import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

import joblib
import pandas as pd
import os
from datetime import datetime


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
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ML_DIR = os.path.join(
    BASE_DIR,
    "ml"
)


# =====================================================
# MODEL PATHS
# =====================================================

CATEGORY_MODEL_PATH = os.path.join(
    ML_DIR,
    "aqi_category_model.pkl"
)

FORECAST_MODEL_PATH = os.path.join(
    ML_DIR,
    "future_aqi_models.pkl"
)


# =====================================================
# LOAD AQI CATEGORY MODEL
# =====================================================

try:

    category_model = joblib.load(
        CATEGORY_MODEL_PATH
    )

    print(
        "AQI category ML model loaded successfully."
    )

except FileNotFoundError:

    print(
        "ERROR: aqi_category_model.pkl not found."
    )

    print(
        "Expected path:",
        CATEGORY_MODEL_PATH
    )

    category_model = None

except Exception as e:

    print(
        "ERROR loading AQI category model:",
        repr(e)
    )

    category_model = None


# =====================================================
# LOAD FUTURE AQI MODELS
# =====================================================

try:

    forecast_models = joblib.load(
        FORECAST_MODEL_PATH
    )

    model_1h = forecast_models["model_1h"]
    model_2h = forecast_models["model_2h"]
    model_3h = forecast_models["model_3h"]

    FORECAST_FEATURES = forecast_models[
        "features"
    ]

    print(
        "Future AQI ML models loaded successfully."
    )

    print(
        "Forecast features:",
        FORECAST_FEATURES
    )

except FileNotFoundError:

    print(
        "ERROR: future_aqi_models.pkl not found."
    )

    print(
        "Expected path:",
        FORECAST_MODEL_PATH
    )

    model_1h = None
    model_2h = None
    model_3h = None

    FORECAST_FEATURES = [
        "aqi",
        "pm25",
        "pm10",
        "rel_humidity",
        "temperature",
        "hour",
        "month",
        "latitude",
        "longitude"
    ]

except Exception as e:

    print(
        "ERROR loading future AQI models:",
        repr(e)
    )

    model_1h = None
    model_2h = None
    model_3h = None

    FORECAST_FEATURES = [
        "aqi",
        "pm25",
        "pm10",
        "rel_humidity",
        "temperature",
        "hour",
        "month",
        "latitude",
        "longitude"
    ]


# =====================================================
# AQI CATEGORY FALLBACK
# =====================================================

def get_aqi_category(aqi):

    aqi = float(aqi)

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
# AI HEALTH ADVICE
# =====================================================

def get_advice(aqi):

    aqi = float(aqi)

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
# PRESENT AQI CATEGORY USING ML
# =====================================================

def classify_present_aqi(present_aqi):

    present_aqi = round(
        float(present_aqi)
    )

    # -------------------------------------------------
    # Use trained ML classifier when available
    # -------------------------------------------------

    if category_model is not None:

        try:

            category_input = pd.DataFrame([
                {
                    "aqi": present_aqi
                }
            ])

            prediction = category_model.predict(
                category_input
            )[0]

            return str(
                prediction
            )

        except Exception as e:

            print(
                "ML classifier prediction error:",
                repr(e)
            )

    # -------------------------------------------------
    # Fallback
    # -------------------------------------------------

    return get_aqi_category(
        present_aqi
    )


# =====================================================
# FUTURE AQI PREDICTION
# =====================================================

def predict_future_aqi(
    present_aqi,
    pm25,
    pm10,
    humidity,
    temperature,
    latitude,
    longitude
):

    if (
        model_1h is None
        or model_2h is None
        or model_3h is None
    ):

        print(
            "Future models unavailable."
        )

        return []


    try:

        # -------------------------------------------------
        # CURRENT TIME
        # -------------------------------------------------

        now = datetime.now()

        current_hour = now.hour
        current_month = now.month


        # -------------------------------------------------
        # MODEL INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame([
            {

                "aqi":
                    float(present_aqi),

                "pm25":
                    float(pm25),

                "pm10":
                    float(pm10),

                "rel_humidity":
                    float(humidity),

                "temperature":
                    float(temperature),

                "hour":
                    float(current_hour),

                "month":
                    float(current_month),

                "latitude":
                    float(latitude),

                "longitude":
                    float(longitude)

            }
        ])


        # -------------------------------------------------
        # MATCH TRAINING FEATURE ORDER
        # -------------------------------------------------

        input_data = input_data[
            FORECAST_FEATURES
        ]


        # -------------------------------------------------
        # +1 HOUR
        # -------------------------------------------------

        prediction_1h = model_1h.predict(
            input_data
        )[0]

        prediction_1h = max(
            0,
            round(
                float(
                    prediction_1h
                )
            )
        )


        # -------------------------------------------------
        # +2 HOURS
        # -------------------------------------------------

        prediction_2h = model_2h.predict(
            input_data
        )[0]

        prediction_2h = max(
            0,
            round(
                float(
                    prediction_2h
                )
            )
        )


        # -------------------------------------------------
        # +3 HOURS
        # -------------------------------------------------

        prediction_3h = model_3h.predict(
            input_data
        )[0]

        prediction_3h = max(
            0,
            round(
                float(
                    prediction_3h
                )
            )
        )


        # -------------------------------------------------
        # RETURN FORECAST
        # -------------------------------------------------

        return [

            {
                "hours_ahead": 1,
                "aqi": prediction_1h,
                "category":
                    get_aqi_category(
                        prediction_1h
                    )
            },

            {
                "hours_ahead": 2,
                "aqi": prediction_2h,
                "category":
                    get_aqi_category(
                        prediction_2h
                    )
            },

            {
                "hours_ahead": 3,
                "aqi": prediction_3h,
                "category":
                    get_aqi_category(
                        prediction_3h
                    )
            }

        ]


    except Exception as e:

        print(
            "Future AQI prediction error:",
            repr(e)
        )

        return []


# =====================================================
# HOME
# =====================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =====================================================
# FORECAST TEST API
# =====================================================

@app.route(
    "/api/forecast",
    methods=["GET"]
)
def forecast_endpoint():

    try:

        latitude = request.args.get(
            "latitude"
        )

        longitude = request.args.get(
            "longitude"
        )

        present_aqi = request.args.get(
            "aqi"
        )

        pm25 = request.args.get(
            "pm25",
            0
        )

        pm10 = request.args.get(
            "pm10",
            0
        )

        humidity = request.args.get(
            "hum",
            0
        )

        temperature = request.args.get(
            "temp",
            0
        )


        if (
            latitude is None
            or longitude is None
            or present_aqi is None
        ):

            return jsonify({

                "status":
                    "error",

                "message":
                    "latitude, longitude and aqi are required"

            }), 400


        forecast = predict_future_aqi(

            present_aqi=
                present_aqi,

            pm25=
                pm25,

            pm10=
                pm10,

            humidity=
                humidity,

            temperature=
                temperature,

            latitude=
                latitude,

            longitude=
                longitude

        )


        return jsonify({

            "status":
                "success",

            "present_aqi":
                round(
                    float(
                        present_aqi
                    )
                ),

            "future_forecast":
                forecast,

            "forecast_source":
                "AeroSense ML"

        })


    except Exception as e:

        print(
            "Forecast endpoint error:",
            repr(e)
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 400


# =====================================================
# ESP32 DATA UPLOAD
# =====================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_data():

    try:

        # =================================================
        # RECEIVE ESP32 JSON
        # =================================================

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "status":
                    "error",

                "message":
                    "No JSON data received"

            }), 400


        print()
        print(
            "======================================"
        )

        print(
            "RECEIVED FROM ESP32"
        )

        print(
            "======================================"
        )

        print(
            data
        )


        # =================================================
        # PRESENT AQI
        #
        # IMPORTANT:
        # This comes directly from ESP32.
        #
        # We DO NOT calculate another present AQI.
        # =================================================

        if "final_aqi" not in data:

            return jsonify({

                "status":
                    "error",

                "message":
                    "ESP32 final_aqi is missing"

            }), 400


        present_aqi = round(
            float(
                data["final_aqi"]
            )
        )


        # =================================================
        # ML CLASSIFICATION
        #
        # ML understands whether the current AQI is:
        #
        # Good
        # Satisfactory
        # Moderate
        # Poor
        # Very Poor
        # Severe
        # =================================================

        present_category = classify_present_aqi(
            present_aqi
        )


        # =================================================
        # SENSOR VALUES
        # =================================================

        pm25 = float(
            data.get(
                "pm25",
                0
            )
        )

        pm10 = float(
            data.get(
                "pm10",
                0
            )
        )

        humidity = float(
            data.get(
                "hum",
                0
            )
        )

        temperature = float(
            data.get(
                "temp",
                0
            )
        )


        # =================================================
        # GPS
        # =================================================

        gps_fix = (
            data.get(
                "gps_fix",
                False
            )
            is True
        )


        latitude = data.get(
            "latitude"
        )

        longitude = data.get(
            "longitude"
        )

        satellites = data.get(
            "satellites"
        )

        accuracy = data.get(
            "accuracy"
        )

        maps_url = data.get(
            "maps_url",
            ""
        )


        # =================================================
        # FUTURE AQI ML FORECAST
        # =================================================

        future_forecast = []


        if (
            gps_fix
            and latitude is not None
            and longitude is not None
        ):

            future_forecast = predict_future_aqi(

                present_aqi=
                    present_aqi,

                pm25=
                    pm25,

                pm10=
                    pm10,

                humidity=
                    humidity,

                temperature=
                    temperature,

                latitude=
                    latitude,

                longitude=
                    longitude

            )


        # =================================================
        # AI ADVICE
        #
        # Advice is based on PRESENT AQI.
        # =================================================

        advice = get_advice(
            present_aqi
        )


        # =================================================
        # BUILD DASHBOARD DATA
        # =================================================

        dashboard_data = dict(
            data
        )


        # =================================================
        # PRESENT AQI
        # =================================================

        dashboard_data["present_aqi"] = (
            present_aqi
        )

        dashboard_data["present_category"] = (
            present_category
        )


        # =================================================
        # ML CATEGORY
        # =================================================

        dashboard_data["ml_category"] = (
            present_category
        )


        # -------------------------------------------------
        # IMPORTANT:
        # Keep ML prediction field for old dashboard
        # compatibility, BUT it represents the current
        # ESP32 AQI being classified.
        # -------------------------------------------------

        dashboard_data["ml_prediction"] = (
            present_aqi
        )


        # =================================================
        # DASHBOARD GAUGE
        # =================================================

        dashboard_data["final_aqi"] = (
            present_aqi
        )

        dashboard_data["category"] = (
            present_category
        )


        # =================================================
        # ENVIRONMENT
        # =================================================

        dashboard_data["temp"] = (
            temperature
        )

        dashboard_data["hum"] = (
            humidity
        )


        # =================================================
        # DOMINANT SIGNAL
        # =================================================

        dashboard_data["dominant"] = (
            "ESP32 AQI"
        )


        # =================================================
        # ADVICE
        # =================================================

        dashboard_data["advice"] = (
            advice
        )


        # =================================================
        # GPS
        # =================================================

        dashboard_data["gps_fix"] = (
            gps_fix
        )

        dashboard_data["latitude"] = (
            latitude
        )

        dashboard_data["longitude"] = (
            longitude
        )

        dashboard_data["satellites"] = (
            satellites
        )

        dashboard_data["accuracy"] = (
            accuracy
        )

        dashboard_data["maps_url"] = (
            maps_url
        )


        # =================================================
        # FUTURE FORECAST
        # =================================================

        dashboard_data["future_forecast"] = (
            future_forecast
        )

        dashboard_data["forecast_source"] = (
            "AeroSense ML"
        )


        # =================================================
        # SEND TO DASHBOARD
        # =================================================

        socketio.emit(
            "live_data",
            dashboard_data
        )


        # =================================================
        # SERVER LOGS
        # =================================================

        print()
        print(
            "------------------------------"
        )

        print(
            "PRESENT AQI:",
            present_aqi
        )

        print(
            "PRESENT CATEGORY:",
            present_category
        )

        print(
            "AI ADVICE:",
            advice
        )

        print(
            "GPS FIX:",
            gps_fix
        )

        print(
            "LATITUDE:",
            latitude
        )

        print(
            "LONGITUDE:",
            longitude
        )

        print(
            "SATELLITES:",
            satellites
        )

        print(
            "ACCURACY:",
            accuracy
        )


        print(
            "FUTURE AQI FORECAST:"
        )


        if future_forecast:

            for item in future_forecast:

                print(
                    "+",
                    item["hours_ahead"],
                    "hour:",
                    item["aqi"],
                    item["category"]
                )

        else:

            print(
                "Future forecast unavailable."
            )


        print(
            "------------------------------"
        )


        # =================================================
        # RESPONSE TO ESP32
        # =================================================

        return jsonify({

            "status":
                "success",

            "message":
                "Present AQI classified and future AQI predicted",


            # -------------------------------------------------
            # PRESENT AQI
            # -------------------------------------------------

            "present_aqi":
                present_aqi,

            "present_category":
                present_category,


            # -------------------------------------------------
            # ML CLASSIFICATION
            # -------------------------------------------------

            "ml_prediction":
                present_aqi,

            "ml_category":
                present_category,


            # -------------------------------------------------
            # DASHBOARD COMPATIBILITY
            # -------------------------------------------------

            "final_aqi":
                present_aqi,

            "category":
                present_category,


            # -------------------------------------------------
            # ADVICE
            # -------------------------------------------------

            "advice":
                advice,


            # -------------------------------------------------
            # FUTURE AQI
            # -------------------------------------------------

            "predicted_1h":
                (
                    future_forecast[0]["aqi"]
                    if len(future_forecast) > 0
                    else None
                ),

            "predicted_2h":
                (
                    future_forecast[1]["aqi"]
                    if len(future_forecast) > 1
                    else None
                ),

            "predicted_3h":
                (
                    future_forecast[2]["aqi"]
                    if len(future_forecast) > 2
                    else None
                ),

            "future_forecast":
                future_forecast,

            "forecast_source":
                "AeroSense ML",


            # -------------------------------------------------
            # GPS
            # -------------------------------------------------

            "gps_fix":
                gps_fix,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "satellites":
                satellites,

            "accuracy":
                accuracy,

            "maps_url":
                maps_url,


            # -------------------------------------------------
            # FULL DASHBOARD DATA
            # -------------------------------------------------

            "data":
                dashboard_data

        })


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print()
        print(
            "======================================"
        )

        print(
            "UPLOAD ERROR"
        )

        print(
            "======================================"
        )

        print(
            repr(e)
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
