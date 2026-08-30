import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# 1. LOAD CSV
# =====================================================

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "west_bengal_aqi_cleaned.csv"
)

df = pd.read_csv(CSV_PATH)

print("Dataset loaded.")
print(df.head())
print(df.columns.tolist())

# =====================================================
# 2. FEATURES
# =====================================================

FEATURES = [
    "pm25",
    "pm10",
    "rel_humidity",
    "temperature"
]

TARGET = "aqi"

X = df[FEATURES]
y = df[TARGET]

# =====================================================
# 3. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 4. TRAIN MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# 5. TEST
# =====================================================

score = model.score(X_test, y_test)

print("Model R²:", score)

# =====================================================
# 6. SAVE MODEL
# =====================================================

os.makedirs("ml", exist_ok=True)

MODEL_PATH = os.path.join(
    "ml",
    "aerosense_aqi_model.pkl"
)

joblib.dump(
    model,
    MODEL_PATH,
    protocol=5
)

print("Model saved:", MODEL_PATH)

# =====================================================
# 7. VERIFY MODEL
# =====================================================

test_model = joblib.load(MODEL_PATH)

print("Model reload successful.")

test_input = pd.DataFrame([{
    "pm25": 76.4,
    "pm10": 112,
    "rel_humidity": 68,
    "temperature": 31.2
}])

prediction = test_model.predict(test_input)[0]

print("Test AQI prediction:", round(float(prediction)))