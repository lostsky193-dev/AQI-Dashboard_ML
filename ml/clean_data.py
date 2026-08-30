import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================

input_file = "ml/data/west_bengal_aqi.csv"
output_file = "ml/data/west_bengal_aqi_cleaned.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# ==========================================
# 3. CONVERT DATE
# ==========================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove rows where date could not be converted
df = df.dropna(subset=["date"])


# ==========================================
# 4. CHECK NUMERICAL COLUMNS
# ==========================================

numeric_columns = [
    "aqi",
    "pm25",
    "pm10",
    "rel_humidity",
    "temperature"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ==========================================
# 5. REMOVE INVALID VALUES
# ==========================================

# AQI cannot be negative
df = df[df["aqi"] >= 0]

# PM values cannot be negative
df = df[df["pm25"] >= 0]
df = df[df["pm10"] >= 0]

# Humidity should be between 0 and 100
df = df[
    (df["rel_humidity"] >= 0) &
    (df["rel_humidity"] <= 100)
]


# ==========================================
# 6. REMOVE MISSING VALUES CREATED ABOVE
# ==========================================

df = df.dropna(subset=numeric_columns)


# ==========================================
# 7. SORT DATA
# ==========================================

df = df.sort_values(
    by=["location", "date", "hour"]
).reset_index(drop=True)


# ==========================================
# 8. SAVE CLEAN DATA
# ==========================================

df.to_csv(output_file, index=False)

print("\n===== CLEANING COMPLETE =====")
print("Final shape:", df.shape)

print("\nRemaining missing values:")
print(df[numeric_columns].isnull().sum())

print("\nFinal columns:")
print(df.columns.tolist())

print("\nSaved to:")
print(output_file)