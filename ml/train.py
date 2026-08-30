import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD CLEAN DATA
# ==========================================

file_path = "ml/data/west_bengal_aqi_cleaned.csv"

df = pd.read_csv(file_path)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

features = [
    "pm25",
    "pm10",
    "rel_humidity",
    "temperature"
]

target = "aqi"

X = df[features]
y = df[target]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 4. CREATE RANDOM FOREST
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. TRAIN
# ==========================================

print("Training Random Forest...")

model.fit(X_train, y_train)


# ==========================================
# 6. PREDICT
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. EVALUATE
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n======================================")
print("       RANDOM FOREST RESULTS")
print("======================================")

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ==========================================
# 8. SAVE MODEL
# ==========================================

model_path = "ml/aerosense_aqi_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully:")
print(model_path)


# ==========================================
# 9. SAVE FEATURE ORDER
# ==========================================

feature_path = "ml/model_features.txt"

with open(feature_path, "w") as f:
    for feature in features:
        f.write(feature + "\n")

print("\nFeature order saved:")
print(feature_path)