from pathlib import Path
import time
import pandas as pd

START_YEAR = 2018
END_YEAR = 2026
SLEEP_SECONDS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

for year in range(START_YEAR, END_YEAR + 1):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    out_path = RAW_DIR / f"NBA_{year}_per_game.csv"

    if out_path.exists():
        print(f"SKIP {year}: {out_path.name} already exists")
        continue

    print(f"Fetching {year} per-game...")
    tables = pd.read_html(url)
    df = tables[0]

    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str) != "Rk"].copy()

    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.name}")

    time.sleep(SLEEP_SECONDS)

print("DONE")
