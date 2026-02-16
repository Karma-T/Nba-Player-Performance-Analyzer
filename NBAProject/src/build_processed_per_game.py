from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def infer_year_from_filename(name: str) -> int:
    # expects NBA_YYYY_per_game.csv
    return int(name.split("_")[1])

all_dfs = []

for csv_path in sorted(RAW_DIR.glob("NBA_*_per_game.csv")):
    year = infer_year_from_filename(csv_path.name)

    df = pd.read_csv(csv_path)

    # Remove repeated header rows if they somehow exist
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str) != "Rk"].copy()

    df["season_year"] = year

    all_dfs.append(df)

big = pd.concat(all_dfs, ignore_index=True)

# Clean traded player duplicates:
# Keep TOT row if it exists for that player-season, otherwise keep all rows
if "Player" in big.columns and "Tm" in big.columns:
    has_tot = big.groupby(["Player", "season_year"])["Tm"].transform(lambda x: (x == "TOT").any())
    big = big[(~has_tot) | (big["Tm"] == "TOT")].copy()

# Convert numeric columns where possible
for col in big.columns:
    if col in ["Player", "Pos", "Tm"]:
        continue
    big[col] = pd.to_numeric(big[col], errors="coerce")

out_path = OUT_DIR / "nba_per_game_2018_2026_processed.csv"
big.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(big), "Cols:", len(big.columns))
print(big.head(5))
