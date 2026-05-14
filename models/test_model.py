# models/test_model.py
"""
Quick sanity check — predicts scores for a sample roster
against a known pitcher and verifies the rankings make sense.
"""

import pandas as pd

from db.connection import get_engine
from models.train_model import load_model, predict_lineup_scores


def main():
    engine = get_engine()

    query = """
        SELECT player_id, full_name, bats, throws
        FROM players
        WHERE full_name IN (
            'Judge, Aaron', 'Ohtani, Shohei', 'Betts, Mookie',
            'Freeman, Freddie', 'Alvarez, Yordan',
            'Soto, Juan', 'Trout, Mike', 'Goldschmidt, Paul'
        )
    """
    players = pd.read_sql(query, engine)
    print("Players found:")
    print(players.to_string())

    pitcher_query = """
        SELECT player_id, full_name, throws
        FROM players
        WHERE full_name ILIKE '%skenes%'
    """
    pitcher = pd.read_sql(pitcher_query, engine)
    print("\nPitcher:")
    print(pitcher.to_string())

    if players.empty or pitcher.empty:
        print("No players found — check name formatting")
        return

    pipeline = load_model()
    batter_ids = players["player_id"].tolist()
    pitcher_id = int(pitcher.iloc[0]["player_id"])

    scores = predict_lineup_scores(batter_ids, pitcher_id, pipeline)
    print(f"\nPredicted scores vs {pitcher.iloc[0]['full_name']}:")
    print(scores.to_string())


if __name__ == "__main__":
    main()