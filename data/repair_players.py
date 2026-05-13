# data/repair_players.py
"""
Re-processes all at_bats to repair the players table.
Fixes the ON CONFLICT DO NOTHING bug that left bats/throws split
across separate rows instead of merged onto one row per player.

This script does NOT reload at_bats or pitches — only repairs players.
Runtime: ~5-10 minutes.
"""

import time
import pandas as pd
from datetime import date, timedelta
from pybaseball import statcast
from db.connection import get_connection
import pybaseball
pybaseball.cache.enable()

SEASONS = [
    ("2024-03-20", "2024-09-29"),
    ("2025-03-27", "2025-09-28"),
    ("2026-03-26", "2026-05-03"),
]
WEEK = timedelta(days=7)


def weekly_ranges(season_start, season_end):
    start = date.fromisoformat(season_start)
    end   = date.fromisoformat(season_end)
    while start <= end:
        chunk_end = min(start + WEEK - timedelta(days=1), end)
        yield start.isoformat(), chunk_end.isoformat()
        start += WEEK


def repair_players(df: pd.DataFrame, conn):
    """
    Same as load_players() but uses ON CONFLICT DO UPDATE
    so bats and throws get merged onto the same player row.
    """
    cursor = conn.cursor()

    batters = (
        df[["batter", "player_name", "stand"]]
        .drop_duplicates("batter")
        .rename(columns={"batter": "player_id", "stand": "bats"})
    )
    pitchers = (
        df[["pitcher", "player_name", "p_throws"]]
        .drop_duplicates("pitcher")
        .rename(columns={"pitcher": "player_id", "p_throws": "throws"})
    )

    for _, row in batters.iterrows():
        cursor.execute("""
            INSERT INTO players (player_id, full_name, bats)
            VALUES (%s, %s, %s)
            ON CONFLICT (player_id) DO UPDATE
                SET bats      = EXCLUDED.bats,
                    full_name = EXCLUDED.full_name
        """, (int(row.player_id), str(row.player_name), str(row.bats)))

    for _, row in pitchers.iterrows():
        cursor.execute("""
            INSERT INTO players (player_id, full_name, throws)
            VALUES (%s, %s, %s)
            ON CONFLICT (player_id) DO UPDATE
                SET throws    = EXCLUDED.throws,
                    full_name = EXCLUDED.full_name
        """, (int(row.player_id), str(row.player_name), str(row.throws)))

    conn.commit()
    cursor.close()


if __name__ == "__main__":
    print("Repairing players table...")
    conn = get_connection()
    total_weeks = 0

    for season_start, season_end in SEASONS:
        print(f"\n  Season {season_start[:4]}...")
        for week_start, week_end in weekly_ranges(season_start, season_end):
            print(f"    {week_start} → {week_end} ...", end=" ", flush=True)
            try:
                df = statcast(start_dt=week_start, end_dt=week_end)
                if df is None or df.empty:
                    print("no data")
                    continue
                df = df.dropna(subset=["batter", "pitcher", "game_pk"])
                repair_players(df, conn)
                total_weeks += 1
                print("✅")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️  {e}")
                time.sleep(5)
                continue

    conn.close()
    print(f"\n🎉 Repair complete — processed {total_weeks} weeks")