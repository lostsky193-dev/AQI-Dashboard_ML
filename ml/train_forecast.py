import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "west_bengal_aqi_cleaned.csv"
)

MODEL_DIR = BASE_DIR

FORECAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "future_aqi_models.pkl"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "aqi_category_model.pkl"
)


# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

print("Columns:")
print(df.columns.tolist())


# =====================================================
# PREPARE DATE / TIME
# =====================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day


df["hour"] = pd.to_numeric(
    df["hour"],
    errors="coerce"
)

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)

df["aqi"] = pd.to_numeric(
    df["aqi"],
    errors="coerce"
)

df["pm25"] = pd.to_numeric(
    df["pm25"],
    errors="coerce"
)

df["pm10"] = pd.to_numeric(
    df["pm10"],
    errors="coerce"
)

df["rel_humidity"] = pd.to_numeric(
    df["rel_humidity"],
    errors="coerce"
)

df["temperature"] = pd.to_numeric(
    df["temperature"],
    errors="coerce"
)


# =====================================================
# SORT BY LOCATION + TIME
# =====================================================

df = df.sort_values(
    [
        "latitude",
        "longitude",
        "date",
        "hour"
    ]
).reset_index(drop=True)


# =====================================================
# CREATE FUTURE AQI TARGETS
#
# +1 hour
# +2 hours
# +3 hours
# =====================================================

df["aqi_plus_1h"] = (
    df.groupby(
        ["latitude", "longitude"]
    )["aqi"].shift(-1)
)

df["aqi_plus_2h"] = (
    df.groupby(
        ["latitude", "longitude"]
    )["aqi"].shift(-2)
)

df["aqi_plus_3h"] = (
    df.groupby(
        ["latitude", "longitude"]
    )["aqi"].shift(-3)
)


# =====================================================
# FORECAST FEATURES
# =====================================================

FEATURES = [

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


TARGETS = [

    "aqi_plus_1h",

    "aqi_plus_2h",

    "aqi_plus_3h"

]


# =====================================================
# CLEAN FORECAST DATA
# =====================================================

forecast_df = df.dropna(
    subset=FEATURES + TARGETS
).copy()


print(
    "Forecast training rows:",
    len(forecast_df)
)


X = forecast_df[FEATURES]

y1 = forecast_df[
    "aqi_plus_1h"
]

y2 = forecast_df[
    "aqi_plus_2h"
]

y3 = forecast_df[
    "aqi_plus_3h"
]


# =====================================================
# TRAIN / TEST
# =====================================================

X_train, X_test, y1_train, y1_test = train_test_split(
    X,
    y1,
    test_size=0.2,
    random_state=42
)


_, _, y2_train, y2_test = train_test_split(
    X,
    y2,
    test_size=0.2,
    random_state=42
)


_, _, y3_train, y3_test = train_test_split(
    X,
    y3,
    test_size=0.2,
    random_state=42
)


# =====================================================
# FUTURE AQI MODELS
#
# Optimized for low-memory Render deployment.
#
# 5 trees
# max_depth=15
# min_samples_leaf=5
# n_jobs=1
# =====================================================

model_1h = RandomForestRegressor(

    n_estimators=5,

    max_depth=15,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=1

)


model_2h = RandomForestRegressor(

    n_estimators=5,

    max_depth=15,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=1

)


model_3h = RandomForestRegressor(

    n_estimators=5,

    max_depth=15,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=1

)


# =====================================================
# TRAIN FORECAST MODELS
# =====================================================

print()
print("Training +1 hour model...")

model_1h.fit(
    X_train,
    y1_train
)


print("Training +2 hour model...")

model_2h.fit(
    X_train,
    y2_train
)


print("Training +3 hour model...")

model_3h.fit(
    X_train,
    y3_train
)


# =====================================================
# EVALUATION
# =====================================================

print()
print("Evaluating forecast models...")


pred1 = model_1h.predict(
    X_test
)

pred2 = model_2h.predict(
    X_test
)

pred3 = model_3h.predict(
    X_test
)


mae1 = mean_absolute_error(
    y1_test,
    pred1
)

mae2 = mean_absolute_error(
    y2_test,
    pred2
)

mae3 = mean_absolute_error(
    y3_test,
    pred3
)


print()
print("====================================")
print("FUTURE AQI MODEL RESULTS")
print("====================================")


print(
    "MAE +1 hour:",
    round(mae1, 2)
)


print(
    "MAE +2 hour:",
    round(mae2, 2)
)


print(
    "MAE +3 hour:",
    round(mae3, 2)
)


# =====================================================
# SAVE FORECAST MODELS
#
# Compression level 6.
# =====================================================

forecast_models = {

    "features":
        FEATURES,

    "model_1h":
        model_1h,

    "model_2h":
        model_2h,

    "model_3h":
        model_3h

}


joblib.dump(

    forecast_models,

    FORECAST_MODEL_PATH,

    protocol=5,

    compress=6

)


print()
print(
    "Saved:",
    FORECAST_MODEL_PATH
)


# =====================================================
# PRESENT AQI CATEGORY
# =====================================================

def aqi_category(aqi):

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
# PREPARE CLASSIFIER DATA
# =====================================================

classifier_df = df.dropna(
    subset=["aqi"]
).copy()


classifier_df["category"] = (

    classifier_df["aqi"]
    .apply(aqi_category)

)


X_class = classifier_df[
    ["aqi"]
]


y_class = classifier_df[
    "category"
]


# =====================================================
# CLASSIFIER TRAIN / TEST
# =====================================================

Xc_train, Xc_test, yc_train, yc_test = train_test_split(

    X_class,

    y_class,

    test_size=0.2,

    random_state=42,

    stratify=y_class

)


# =====================================================
# AQI CATEGORY CLASSIFIER
# =====================================================

category_model = RandomForestClassifier(

    n_estimators=50,

    random_state=42,

    n_jobs=1

)


print()
print(
    "Training present AQI category model..."
)


category_model.fit(

    Xc_train,

    yc_train

)


# =====================================================
# CLASSIFIER EVALUATION
# =====================================================

category_pred = category_model.predict(

    Xc_test

)


accuracy = accuracy_score(

    yc_test,

    category_pred

)


print(

    "Category classifier accuracy:",

    round(
        accuracy * 100,
        2
    ),

    "%"

)


# =====================================================
# SAVE CATEGORY MODEL
# =====================================================

joblib.dump(

    category_model,

    CLASSIFIER_PATH,

    protocol=5,

    compress=3

)


print(

    "Saved:",

    CLASSIFIER_PATH

)


# =====================================================
# LOCAL CLASSIFIER TEST
# =====================================================

test_aqi = 33


test_category = category_model.predict(

    pd.DataFrame(
        [
            {
                "aqi": test_aqi
            }
        ]
    )

)[0]


print()
print("====================================")
print("PRESENT AQI TEST")
print("====================================")


print(

    "Current AQI:",

    test_aqi

)


print(

    "Category:",

    test_category

)


# =====================================================
# FUTURE AQI LOCAL TEST
# =====================================================

test_row = forecast_df.iloc[0]


test_input = pd.DataFrame(
    [
        {

            "aqi":
                test_row["aqi"],

            "pm25":
                test_row["pm25"],

            "pm10":
                test_row["pm10"],

            "rel_humidity":
                test_row["rel_humidity"],

            "temperature":
                test_row["temperature"],

            "hour":
                test_row["hour"],

            "month":
                test_row["month"],

            "latitude":
                test_row["latitude"],

            "longitude":
                test_row["longitude"]

        }
    ]
)


future1 = model_1h.predict(

    test_input

)[0]


future2 = model_2h.predict(

    test_input

)[0]


future3 = model_3h.predict(

    test_input

)[0]


print()
print("====================================")
print("FUTURE AQI TEST")
print("====================================")


print(

    "+1 hour:",

    round(
        float(future1)
    )

)


print(

    "+2 hours:",

    round(
        float(future2)
    )

)


print(

    "+3 hours:",

    round(
        float(future3)
    )

)


# =====================================================
# MODEL SIZE
# =====================================================

if os.path.exists(
    FORECAST_MODEL_PATH
):

    model_size_bytes = os.path.getsize(
        FORECAST_MODEL_PATH
    )

    model_size_mb = (
        model_size_bytes
        /
        (1024 * 1024)
    )

    print()
    print(
        "Forecast model size:",
        round(
            model_size_mb,
            2
        ),
        "MB"
    )


if os.path.exists(
    CLASSIFIER_PATH
):

    classifier_size_bytes = os.path.getsize(
        CLASSIFIER_PATH
    )

    classifier_size_mb = (
        classifier_size_bytes
        /
        (1024 * 1024)
    )

    print(
        "Category model size:",
        round(
            classifier_size_mb,
            2
        ),
        "MB"
    )


# =====================================================
# TRAINING COMPLETE
# =====================================================

print()
print("====================================")
print("TRAINING COMPLETE")
print("====================================")