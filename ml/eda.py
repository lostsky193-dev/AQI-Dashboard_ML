import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD CLEAN DATA
# ==========================================

file_path = "ml/data/west_bengal_aqi_cleaned.csv"

df = pd.read_csv(file_path)

print("\n===== DATASET OVERVIEW =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ==========================================
# AQI STATISTICS
# ==========================================

print("\n===== AQI STATISTICS =====")
print(df["aqi"].describe())

# ==========================================
# FEATURE STATISTICS
# ==========================================

print("\n===== FEATURE STATISTICS =====")

features = [
    "pm25",
    "pm10",
    "rel_humidity",
    "temperature"
]

print(df[features].describe())

# ==========================================
# AQI CATEGORY
# ==========================================

def aqi_category(aqi):
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


df["aqi_category"] = df["aqi"].apply(aqi_category)

print("\n===== AQI CATEGORY COUNTS =====")
print(df["aqi_category"].value_counts())

# ==========================================
# CORRELATION
# ==========================================

print("\n===== CORRELATION WITH AQI =====")

correlation = df[
    ["aqi", "pm25", "pm10", "rel_humidity", "temperature"]
].corr()

print(correlation["aqi"].sort_values(ascending=False))

# ==========================================
# AQI DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(df["aqi"], bins=50)

plt.xlabel("AQI")
plt.ylabel("Number of Records")
plt.title("AQI Distribution")

plt.tight_layout()
plt.show()

# ==========================================
# PM2.5 VS AQI
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["pm25"],
    df["aqi"],
    alpha=0.2
)

plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.title("PM2.5 vs AQI")

plt.tight_layout()
plt.show()

# ==========================================
# PM10 VS AQI
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["pm10"],
    df["aqi"],
    alpha=0.2
)

plt.xlabel("PM10")
plt.ylabel("AQI")
plt.title("PM10 vs AQI")

plt.tight_layout()
plt.show()