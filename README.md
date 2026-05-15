# MLB Lineup Optimizer ⚾

🌐 **Live Demo:** https://mlb-lineup-optimizer-nsafaj8rhee8rnmmta7d5m.streamlit.app/

An MLB batting order optimizer that uses machine learning and Monte Carlo simulation to find the optimal 9-man lineup against a specific opposing pitcher.

## How It Works

> ⏱️ **Runtime: ~120 seconds** — the optimizer runs 8,000 Monte Carlo game simulations per lineup evaluation.

**Stage 1 — Player Selection (Gradient Boosting ML)**
Scores each batter's expected run contribution against the opposing pitcher using 16 Statcast features including handedness-specific xwOBA, exit velocity, HR rate, and whiff rate. Trained on 435,000 batter-pitcher matchups from 2024-2026.

The model outputs a **hybrid score** — a 50/50 blend of the ML prediction and a statistical baseline (linear weights run value) — that balances model sophistication with sabermetric grounding. The ML model achieves 58% lift over the pure statistical baseline.

**Stage 2 — Batting Order Optimization (Monte Carlo Simulation)**
Simulates 8,000 full 9-inning games per lineup evaluation. Each simulation draws discrete at-bat outcomes from each batter's empirical event distribution, adjusted for pitcher difficulty via the odds ratio method. Pairwise-swap local search finds the near-optimal batting order.

**Note on variance:** Lineup ordering effects in baseball are small. Minor slot variations between runs are expected — player selection drives most of the optimizer's value.

## Tech Stack

- **Python** — core language
- **PostgreSQL + Supabase** — database for 1.5M+ Statcast pitches
- **pybaseball** — Statcast data pipeline
- **pandas / numpy** — data manipulation
- **scikit-learn** — Gradient Boosting model
- **Streamlit** — interactive web app
- **SQLAlchemy / psycopg2** — database connectivity

## Data

- 2024, 2025, and 2026 (to date) MLB Statcast data
- ~1.5 million pitches across ~410,000 plate appearances
- Source: Baseball Savant via pybaseball

## Author

Ryan Arpin