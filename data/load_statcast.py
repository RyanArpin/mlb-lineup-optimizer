# data/load_statcast.py
"""
Pulls Statcast pitch-level data from Baseball Savant using pybaseball
and loads it into PostgreSQL.

Strategy:
  - Pull data week-by-week to avoid Baseball Savant rate limits
  - For each week, parse and insert players, games, at_bats, pitches
  - Upsert pattern (INSERT ... ON CONFLICT DO NOTHING) so the script
    is safe to re-run without creating duplicate rows

Runtime estimate: ~20-35 minutes for 2+ full seasons.
"""

import time
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pybaseball import statcast
from psycopg2.extras import execute_values
from db.connection import get_connection, get_engine

# Enable pybaseball's local cache — avoids re-downloading data
# if the script gets interrupted and you re-run it
import pybaseball
pybaseball.cache.enable()


# ── Configuration ────────────────────────────────────────────
SEASONS = [
    ("2024-03-20", "2024-09-29"),  # 2024 full season
    ("2025-03-27", "2025-09-28"),  # 2025 full season
    ("2026-03-26", "2026-05-03"),  # 2026 season to date
]
WEEK = timedelta(days=7)  # pull size — don't increase, will timeout


# ── Helper: generate weekly date ranges ──────────────────────
def weekly_ranges(season_start: str, season_end: str):
    """
    Yields (start, end) string pairs for each week in the season.
    Example: ('2024-03-20', '2024-03-26'), ('2024-03-27', '2024-04-02'), ...
    """
    start = date.fromisoformat(season_start)
    end   = date.fromisoformat(season_end)
    while start <= end:
        chunk_end = min(start + WEEK - timedelta(days=1), end)
        yield start.isoformat(), chunk_end.isoformat()
        start += WEEK


# ── Loaders ──────────────────────────────────────────────────

def load_players(df: pd.DataFrame, conn):
    """
    Inserts unique batters and pitchers into the players table.
    Uses ON CONFLICT DO UPDATE to merge bats/throws onto the same row
    when the same player appears as both a batter and a pitcher.
    """
    cursor = conn.cursor()

    # Batters — insert or update the bats column
    batters = (
        df[["batter", "player_name", "stand"]]
        .drop_duplicates("batter")
        .rename(columns={"batter": "player_id", "stand": "bats"})
    )

    # Pitchers — insert or update the throws column
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


def load_games(df: pd.DataFrame, conn):
    """
    Inserts unique games into the games table.
    """
    cursor = conn.cursor()

    games = (
        df[["game_pk", "game_date", "home_team", "away_team"]]
        .drop_duplicates("game_pk")
    )

    for _, row in games.iterrows():
        game_date = pd.to_datetime(row.game_date).date()
        season    = game_date.year

        cursor.execute("""
            INSERT INTO games (game_pk, game_date, season, home_team, away_team)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (game_pk) DO NOTHING
        """, (
            int(row.game_pk),
            game_date,
            season,
            str(row.home_team),
            str(row.away_team)
        ))

    conn.commit()
    cursor.close()


def load_at_bats(df: pd.DataFrame, conn):
    """
    Inserts unique at-bats into the at_bats table.
    Each at-bat is identified by (game_pk, batter, pitcher, at_bat_number).
    Returns a dict mapping that tuple → at_bat_id for use in load_pitches().
    """
    cursor = conn.cursor()

    # One row per at-bat: take the last pitch of each at-bat
    # (it has the final event/outcome)
    ab_cols = [
        "game_pk", "batter", "pitcher", "at_bat_number",
        "pitch_number", "inning", "inning_topbot", "outs_when_up",
        "on_1b", "on_2b", "on_3b", "events"
    ]
    at_bats = (
        df[ab_cols]
        .sort_values(["game_pk", "batter", "pitcher", "at_bat_number", "pitch_number"])
        .drop_duplicates(["game_pk", "batter", "pitcher", "at_bat_number"],
                         keep="last")
    )

    at_bat_map = {}  # (game_pk, batter, pitcher, at_bat_number) → at_bat_id

    for _, row in at_bats.iterrows():
        inning_half = "top" if row.inning_topbot == "Top" else "bot"

        # on_1b/2b/3b are player IDs when occupied, NaN when empty
        on_1b = not pd.isna(row.on_1b)
        on_2b = not pd.isna(row.on_2b)
        on_3b = not pd.isna(row.on_3b)

        event = str(row.events) if not pd.isna(row.events) else None

        cursor.execute("""
            INSERT INTO at_bats
                (game_pk, batter_id, pitcher_id, at_bat_number, inning, inning_half,
                 outs_when_up, on_1b, on_2b, on_3b, event)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (game_pk, batter_id, pitcher_id, at_bat_number)
            DO UPDATE SET
                inning = EXCLUDED.inning,
                inning_half = EXCLUDED.inning_half,
                outs_when_up = EXCLUDED.outs_when_up,
                on_1b = EXCLUDED.on_1b,
                on_2b = EXCLUDED.on_2b,
                on_3b = EXCLUDED.on_3b,
                event = EXCLUDED.event
            RETURNING at_bat_id
        """, (
            int(row.game_pk),
            int(row.batter),
            int(row.pitcher),
            int(row.at_bat_number),
            int(row.inning),
            inning_half,
            int(row.outs_when_up),
            on_1b, on_2b, on_3b,
            event
        ))

        at_bat_id = cursor.fetchone()[0]
        key = (int(row.game_pk), int(row.batter),
               int(row.pitcher), int(row.at_bat_number))
        at_bat_map[key] = at_bat_id

    conn.commit()
    cursor.close()
    return at_bat_map


def load_pitches(df: pd.DataFrame, conn, at_bat_map: dict):
    """
    Bulk-inserts all pitches using pandas .to_sql() for speed.
    Joins in the at_bat_id from the map built in load_at_bats().
    """
    cursor = conn.cursor()

    # Map at_bat_id onto each pitch row
    def get_at_bat_id(row):
        key = (int(row.game_pk), int(row.batter),
               int(row.pitcher), int(row.at_bat_number))
        return at_bat_map.get(key)

    df = df.copy()
    df["at_bat_id"] = df.apply(get_at_bat_id, axis=1)

    # Select only the columns our schema expects
    pitch_cols = {
        "at_bat_id":                           "at_bat_id",
        "game_pk":                             "game_pk",
        "batter":                              "batter_id",
        "pitcher":                             "pitcher_id",
        "at_bat_number":                       "at_bat_number",
        "pitch_number":                        "pitch_number",
        "game_date":                           "game_date",
        "pitch_type":                          "pitch_type",
        "release_speed":                       "release_speed",
        "release_spin_rate":                   "release_spin_rate",
        "pfx_x":                               "pfx_x",
        "pfx_z":                               "pfx_z",
        "plate_x":                             "plate_x",
        "plate_z":                             "plate_z",
        "zone":                                "zone",
        "description":                         "description",
        "type":                                "type",
        "balls":                               "balls",
        "strikes":                             "strikes",
        "launch_speed":                        "launch_speed",
        "launch_angle":                        "launch_angle",
        "hit_distance_sc":                     "hit_distance_sc",
        "bb_type":                             "bb_type",
        "estimated_ba_using_speedangle":       "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle":     "estimated_woba_using_speedangle",
        "stand":                               "stand",
        "p_throws":                            "p_throws",
        "if_fielding_alignment":               "if_fielding_alignment",
        "of_fielding_alignment":               "of_fielding_alignment",
    }

    # Keep only columns that exist in this pull (some may be missing)
    available = {k: v for k, v in pitch_cols.items() if k in df.columns}
    pitches_df = df[list(available.keys())].rename(columns=available)

    # Replace NaN with None for clean NULL insertion
    pitches_df = pitches_df.where(pd.notna(pitches_df), None)

    insert_cols = [
        "at_bat_id", "game_pk", "batter_id", "pitcher_id",
        "at_bat_number", "pitch_number", "game_date", "pitch_type",
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "zone", "description", "type",
        "balls", "strikes", "launch_speed", "launch_angle",
        "hit_distance_sc", "bb_type", "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle", "stand", "p_throws",
        "if_fielding_alignment", "of_fielding_alignment"
    ]

    pitches_df = pitches_df.reindex(columns=insert_cols)
    rows = [tuple(row) for row in pitches_df.itertuples(index=False, name=None)]

    if rows:
        execute_values(
            cursor,
            """
            INSERT INTO pitches (
                at_bat_id, game_pk, batter_id, pitcher_id,
                at_bat_number, pitch_number, game_date, pitch_type,
                release_speed, release_spin_rate, pfx_x, pfx_z,
                plate_x, plate_z, zone, description, type,
                balls, strikes, launch_speed, launch_angle,
                hit_distance_sc, bb_type, estimated_ba_using_speedangle,
                estimated_woba_using_speedangle, stand, p_throws,
                if_fielding_alignment, of_fielding_alignment
            ) VALUES %s
            ON CONFLICT (game_pk, batter_id, pitcher_id, at_bat_number, pitch_number)
            DO NOTHING
            """,
            rows,
            page_size=1000
        )
        conn.commit()

    cursor.close()


# ── Main orchestrator ─────────────────────────────────────────

def load_season(season_start: str, season_end: str):
    """
    Pulls and loads one full season week by week.
    """
    print(f"\n{'='*55}")
    print(f"  Loading season {season_start[:4]}")
    print(f"{'='*55}")

    conn = get_connection()
    total_pitches = 0

    for week_start, week_end in weekly_ranges(season_start, season_end):
        print(f"  Pulling {week_start} → {week_end} ...", end=" ", flush=True)

        try:
            df = statcast(start_dt=week_start, end_dt=week_end)

            if df is None or df.empty:
                print("no data")
                continue

            # Drop rows missing essential IDs
            df = df.dropna(subset=["batter", "pitcher", "game_pk", "at_bat_number", "pitch_number"])

            load_players(df, conn)
            load_games(df, conn)
            at_bat_map = load_at_bats(df, conn)
            load_pitches(df, conn, at_bat_map)

            total_pitches += len(df)
            print(f"{len(df):,} pitches loaded")

            # Be polite to Baseball Savant's API
            time.sleep(3)

        except Exception as e:
            print(f"\n  ⚠️  Error on {week_start}: {e}")
            print("     Skipping this week and continuing...")
            time.sleep(5)
            continue

    conn.close()
    print(f"\n  ✅ Season complete — {total_pitches:,} total pitches loaded")


if __name__ == "__main__":
    for season_start, season_end in SEASONS:
        load_season(season_start, season_end)

    print("\n🎉 All seasons loaded successfully!")