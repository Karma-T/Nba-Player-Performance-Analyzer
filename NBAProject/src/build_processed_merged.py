from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2018
END_YEAR = 2026

def infer_year(name: str) -> int:
    # NBA_YYYY_per_game.csv or NBA_YYYY_advanced.csv
    return int(name.split("_")[1])

def keep_tot(df: pd.DataFrame, year_col: str = "season_year") -> pd.DataFrame:
    if "Player" in df.columns and "Tm" in df.columns and year_col in df.columns:
        has_tot = df.groupby(["Player", year_col])["Tm"].transform(lambda x: (x == "TOT").any())
        return df[(~has_tot) | (df["Tm"] == "TOT")].copy()

    # If no Tm column exists, we can’t do TOT filtering here; just return as-is
    return df


def clean_br(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flatten multi-index columns if they appear
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Remove repeated header rows
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str).str.strip() != "Rk"].copy()

    df = df.dropna(how="all")
    return df

# -------- Load per-game (from raw files to ensure consistency) --------
per_game_dfs = []
for p in sorted(RAW_DIR.glob("NBA_*_per_game.csv")):
    year = infer_year(p.name)
    if year < START_YEAR or year > END_YEAR:
        continue
    df = pd.read_csv(p)
    df = clean_br(df)
    df["season_year"] = year
    per_game_dfs.append(df)

per_game = pd.concat(per_game_dfs, ignore_index=True)
per_game = keep_tot(per_game)

# -------- Load advanced --------
adv_dfs = []
for p in sorted(RAW_DIR.glob("NBA_*_advanced.csv")):
    year = infer_year(p.name)
    if year < START_YEAR or year > END_YEAR:
        continue
    df = pd.read_csv(p)
    df = clean_br(df)
    df["season_year"] = year
    adv_dfs.append(df)

advanced = pd.concat(adv_dfs, ignore_index=True)
advanced = keep_tot(advanced)

# -------- Merge keys (safe + practical) --------
# We merge on Player + season_year + Tm.
# This works well because we kept TOT (single row per player-season).
merge_keys = ["Player", "season_year"]

merged = per_game.merge(
    advanced,
    on=merge_keys,
    how="left",
    suffixes=("", "_adv")
)

# Convert numeric columns where possible (pandas 3.x safe)
for col in merged.columns:
    if col in ["Player", "Pos", "Tm"]:
        continue
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

out_path = OUT_DIR / f"nba_player_seasons_{START_YEAR}_{END_YEAR}.csv"
merged.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(merged), "Cols:", len(merged.columns))
print(merged.head(3))
