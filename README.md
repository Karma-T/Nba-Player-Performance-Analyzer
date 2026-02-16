# NBA 2027 Player Performance Projection Model

## Overview
This project develops a regression-based predictive model to forecast 2027 NBA player performance using historical season data from 2018–2026.

The model predicts key performance metrics and exports projections for interactive comparison within a Tableau dashboard.

---

## Objectives

- Forecast 2027 player statistics using historical performance trends
- Compare 2026 actual stats vs 2027 projected stats
- Visualize year-over-year changes using Tableau

---

## Data

- Source: Aggregated NBA season data (2018–2026)
- Features include:
  - Age
  - Minutes Played (MP)
  - Points (PTS)
  - Rebounds (TRB)
  - Assists (AST)
  - Usage Rate (USG%)
  - True Shooting % (TS%)
  - Box Plus/Minus (BPM)

Missing values were handled using median imputation.

---

## Modeling Approach

- Model: Linear Regression (scikit-learn)
- Pipeline:
  - Feature selection
  - Median imputation
  - Regression fitting
  - Season-forward prediction

Predictions include:
- PTS
- AST
- TRB
- STL
- BLK
- FG%
- 3PM
- 3PA
- Derived 3P%

---

## Results

- Generated projected 2027 player statistics
- Identified projected scoring increases among high-usage guards
- Visualized projected improvement and decline trends

Interactive Tableau dashboard available here:

🔗 **[Tableau Dashboard Link]**

---

## Project Structure

