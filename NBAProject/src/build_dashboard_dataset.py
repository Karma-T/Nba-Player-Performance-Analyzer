from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

hist_path = PROCESSED_DIR / "nba_player_seasons_2018_2026.csv"
pred_path = PROCESSED_DIR / "nba_predictions_2027.csv"

hist = pd.read_csv(hist_path)
pred = pd.read_csv(pred_path)

# -------------------------
# Historical data (keep context)
# -------------------------
hist_keep = [
    "Player", "season_year",
    "Age", "MP", "USG%",
    "PTS", "TRB", "AST", "STL", "BLK",
    "FG%", "3P%", "FT%"
]

hist = hist[[c for c in hist_keep if c in hist.columns]].copy()
hist["type"] = "Actual"

# -------------------------
# Predictions (rename + align)
# -------------------------
pred = pred.rename(columns={
    "PTS_pred": "PTS",
    "TRB_pred": "TRB",
    "AST_pred": "AST",
    "STL_pred": "STL",
    "BLK_pred": "BLK",
    "FG%_pred": "FG%",
    "3P%_pred": "3P%",
    "FT%_pred": "FT%",
})

pred["season_year"] = pred["predicted_season"]
pred["type"] = "Predicted"

# Add context columns to predictions if missing
for col in ["Age", "MP", "USG%"]:
    if col not in pred.columns and col in hist.columns:
        # carry last known value forward
        last_vals = (
            hist.sort_values("season_year")
            .groupby("Player")[col]
            .last()
        )
        pred[col] = pred["Player"].map(last_vals)

pred_keep = hist.columns.intersection(pred.columns)
pred = pred[pred_keep].copy()

# -------------------------
# Combine
# -------------------------
dashboard_df = pd.concat([hist, pred], ignore_index=True)

out_path = PROCESSED_DIR / "nba_dashboard_dataset.csv"
dashboard_df.to_csv(out_path, index=False)

print("Saved dashboard dataset:", out_path)
print("Columns:", dashboard_df.columns.tolist())
print("Predicted rows:", (dashboard_df["type"] == "Predicted").sum())
