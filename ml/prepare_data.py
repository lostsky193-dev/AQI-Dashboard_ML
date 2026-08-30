import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# LOAD CLEAN DATA
# ==========================================

file_path = "ml/data/west_bengal_aqi_cleaned.csv"

df = pd.read_csv(file_path)

# ==========================================
# SELECT FEATURES AND TARGET
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
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===== ML FEATURES =====")
print(features)

print("\n===== TARGET =====")
print(target)

print("\n===== DATASET SIZE =====")
print("Total samples:", len(df))

print("\n===== TRAINING DATA =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n===== TEST DATA =====")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\n===== SAMPLE TRAINING DATA =====")
print(X_train.head())

print("\n===== SAMPLE TARGET VALUES =====")
print(y_train.head())