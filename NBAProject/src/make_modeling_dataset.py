from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "data" / "processed" / "nba_player_seasons_2018_2026.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_PATH)

# Sort so shifting works correctly
df = df.sort_values(["Player", "season_year"]).copy()

# Stats you want to predict
TARGETS = [
    "PTS", "TRB", "AST", "STL", "BLK",
    "FG%", "3P%", "FT%"
]

# Create next-season targets
for stat in TARGETS:
    if stat in df.columns:
        df[stat.lower().replace("%", "_pct") + "_next"] = (
            df.groupby("Player")[stat].shift(-1)
        )

# Drop rows where we don’t know the next season (last season per player)
next_cols = [c for c in df.columns if c.endswith("_next")]
df_model = df.dropna(subset=next_cols, how="all").copy()

out_path = OUT_DIR / "nba_player_modeling_2018_2026.csv"
df_model.to_csv(out_path, index=False)

print("Saved modeling dataset:", out_path)
print("Rows:", len(df_model))
print("Columns:", len(df_model.columns))
print(df_model[[ "Player", "season_year"] + next_cols].head())
