from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nba_player_modeling_2018_2026.csv"

# Load data
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Target: next-season PTS
# -----------------------------
TARGET = "pts_next"

# Basic feature set (simple but strong baseline)
FEATURES = [
    "PTS", "MP", "G", "Age",
    "TRB", "AST",
    "USG%", "TS%", "BPM"
]

# Keep only available columns
FEATURES = [f for f in FEATURES if f in df.columns]

# Drop rows with missing target or features
df_model = df.dropna(subset=[TARGET] + FEATURES).copy()

# -----------------------------
# Time-aware split
# Train: seasons <= 2022
# Test: seasons >= 2023
# -----------------------------
train = df_model[df_model["season_year"] <= 2022]
test  = df_model[df_model["season_year"] >= 2023]

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("PTS Prediction Model")
print("--------------------")
print("Features:", FEATURES)
print(f"Test MAE:  {mae:.2f} PPG")
print(f"Test RMSE: {rmse:.2f} PPG")

# -----------------------------
# Show sample predictions
# -----------------------------
results = test[["Player", "season_year", "PTS"]].copy()
results["PTS_next_actual"] = y_test.values
results["PTS_next_pred"] = y_pred

print("\nSample predictions:")
print(results.head(10))
