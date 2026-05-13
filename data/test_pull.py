# data/test_pull.py
"""
Pulls one week of Statcast data and prints a summary.
Run this before the full load to verify pybaseball is working.
"""

from pybaseball import statcast
import pybaseball
pybaseball.cache.enable()

print("Pulling one week of Statcast data...")
df = statcast(start_dt="2026-04-01", end_dt="2026-04-07")

print(f"\n✅ Pull successful!")
print(f"   Rows (pitches):  {len(df):,}")
print(f"   Columns:         {len(df.columns)}")
print(f"   Date range:      {df.game_date.min()} → {df.game_date.max()}")
print(f"\nSample columns:\n{list(df.columns[:15])}")
print(f"\nSample row:\n{df.iloc[0][['game_date','player_name','pitch_type','release_speed','events']]}")