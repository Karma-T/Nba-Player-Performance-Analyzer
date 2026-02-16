from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "nba_player_seasons_2018_2026.csv"
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    latest_season = int(df["season_year"].max())
    predicted_season = latest_season + 1

    future_input = df[df["season_year"] == latest_season].copy()

    print(f"Using season {latest_season} to predict season {predicted_season}")
    print("Players in latest season:", future_input["Player"].nunique())

    # Stats to predict (we predict these columns)
    stat_cols = {
        "PTS": "PTS",
        "TRB": "TRB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "FG%": "FG%",
        "3PM": "3PM",
        "3PA": "3PA",
        "FT%": "FT%",
    }

    feature_pool = [
        "PTS", "TRB", "AST", "STL", "BLK",
        "MP", "G", "Age",
        "USG%", "TS%", "BPM"
    ]

    # Include actual 3P% in output if it exists (for comparison in Tableau)
    out_cols = [c for c in ["Player", "Age", "3P%"] if c in future_input.columns]
    predictions = future_input[out_cols].copy()
    predictions["predicted_season"] = predicted_season

    for display_name, y_col in stat_cols.items():
        if y_col not in df.columns:
            print(f"Skipping {display_name}: missing '{y_col}'")
            continue

        # Training rows must have y (stat). Features can be missing -> imputer handles.
        train = df.dropna(subset=[y_col]).copy()

        # Use only features that exist AND have at least SOME non-missing values in training
        features = [f for f in feature_pool if f in df.columns and train[f].notna().any()]

        if len(features) == 0:
            print(f"Skipping {display_name}: no usable features")
            continue

        X_train = train[features]
        y_train = train[y_col]

        X_future = future_input[features]

        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", LinearRegression())
        ])

        model.fit(X_train, y_train)
        predictions[f"{display_name}_pred"] = model.predict(X_future)

        print(f"{display_name}: predicted for {predictions['Player'].nunique()} players using {len(features)} features")

    # -----------------------------
    # Derive predicted 3P% AFTER loop (once)
    # Tableau-friendly name: 3P_pct_pred (no % sign)
    # -----------------------------
    if "3PM_pred" in predictions.columns and "3PA_pred" in predictions.columns:
        # Clean up negatives just in case regression outputs them
        predictions["3PM_pred"] = predictions["3PM_pred"].clip(lower=0)
        predictions["3PA_pred"] = predictions["3PA_pred"].clip(lower=0)

        # Avoid divide-by-zero → set to 0 instead of NULL/blank
        predictions["3P_pct_pred"] = (
            predictions["3PM_pred"] / predictions["3PA_pred"].where(predictions["3PA_pred"] > 0)
        ).fillna(0)

        # Optional: keep within 0..1
        predictions["3P_pct_pred"] = predictions["3P_pct_pred"].clip(0, 1)

    out_path = out_dir / f"nba_predictions_{predicted_season}.csv"
    predictions.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Unique players predicted:", predictions["Player"].nunique())
    print("Prediction columns:", [c for c in predictions.columns if c.endswith("_pred")])


if __name__ == "__main__":
    main()
