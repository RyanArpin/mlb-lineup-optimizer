# data/repair_player_names.py
"""
Fixes player names in the players table using pybaseball's batting
and pitching stats, which correctly map MLBAM IDs to player names.
"""

import pandas as pd
from pybaseball import batting_stats, pitching_stats
from db.connection import get_connection, get_engine
import pybaseball
pybaseball.cache.enable()


def repair_names():
    conn = get_connection()
    cursor = conn.cursor()

    print("Fetching batter names from pybaseball...")
    # Pull batting stats for each season — these have correct name/ID mapping
    batter_frames = []
    for year in [2024, 2025, 2026]:
        try:
            df = batting_stats(year, qual=1)  # qual=1 = minimum 1 PA
            if df is not None and not df.empty:
                batter_frames.append(df[['IDfg', 'Name']].copy())
                print(f"  {year}: {len(df)} batters")
        except Exception as e:
            print(f"  {year}: skipped ({e})")

    print("\nFetching pitcher names from pybaseball...")
    pitcher_frames = []
    for year in [2024, 2025, 2026]:
        try:
            df = pitching_stats(year, qual=1)
            if df is not None and not df.empty:
                pitcher_frames.append(df[['IDfg', 'Name']].copy())
                print(f"  {year}: {len(df)} pitchers")
        except Exception as e:
            print(f"  {year}: skipped ({e})")

    # These use FanGraphs IDs, not MLBAM IDs — we need to cross-reference
    # Instead, let's use the playerid_reverse_lookup which takes MLBAM IDs
    from pybaseball import playerid_reverse_lookup

    print("\nLooking up all player IDs in our database...")
    # Get all player IDs from our database
    from db.connection import get_engine
    engine = get_engine()
    result = pd.read_sql("SELECT player_id FROM players", engine)
    player_ids = result['player_id'].tolist()

    print(f"  Looking up {len(player_ids)} players...")

    # pybaseball can look up names from MLBAM IDs in batches
    lookup = playerid_reverse_lookup(player_ids, key_type='mlbam')

    if lookup is None or lookup.empty:
        print("❌ Lookup returned no results")
        return

    print(f"  Found {len(lookup)} players")

    # Build name from first + last name columns
    lookup['full_name'] = (
        lookup['name_last'].str.capitalize() + ', ' +
        lookup['name_first'].str.capitalize()
    )

    # Update each player's name in the database
    updated = 0
    for _, row in lookup.iterrows():
        cursor.execute("""
            UPDATE players
            SET full_name = %s
            WHERE player_id = %s
        """, (row['full_name'], int(row['key_mlbam'])))
        updated += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Updated {updated} player names")


if __name__ == "__main__":
    repair_names()