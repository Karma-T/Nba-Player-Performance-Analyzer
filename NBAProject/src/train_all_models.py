from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nba_player_modeling_2018_2026.csv"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Stats to predict
# -----------------------------
TARGETS = {
    "PTS": "pts_next",
    "TRB": "trb_next",
    "AST": "ast_next",
    "STL": "stl_next",
    "BLK": "blk_next",
    "FG%": "fg_pct_next",
    "3P%": "fg3_pct_next",
    "FT%": "ft_pct_next",
}

# Shared feature pool
FEATURE_POOL = [
    "PTS", "TRB", "AST", "STL", "BLK",
    "MP", "G", "Age",
    "USG%", "TS%", "BPM"
]

# Time-based split
TRAIN_MAX_YEAR = 2022

print("NBA Player Stat Prediction Models")
print("================================\n")

for stat, target in TARGETS.items():

    if target not in df.columns:
        print(f"Skipping {stat} (missing target)")
        continue

    features = [f for f in FEATURE_POOL if f in df.columns]

    data = df.dropna(subset=[target] + features).copy()

    train = data[data["season_year"] <= TRAIN_MAX_YEAR]
    test  = data[data["season_year"] >  TRAIN_MAX_YEAR]

    if len(train) < 100 or len(test) < 50:
        print(f"{stat}: not enough data, skipped\n")
        continue

    X_train = train[features]
    y_train = train[target]
    X_test  = test[features]
    y_test  = test[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{stat} MODEL")
    print("-" * 30)
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}\n")
