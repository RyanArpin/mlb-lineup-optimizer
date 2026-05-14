# data/migrate_to_supabase.py
"""
Migrates local PostgreSQL data to Supabase.
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text

from db.connection import get_engine as get_local_engine

# ── Supabase connection ───────────────────────────────────────
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_PORT = os.getenv("SUPABASE_PORT", "5432")
SUPABASE_DB = os.getenv("SUPABASE_DB", "postgres")
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")

def get_supabase_engine():
    if not all([SUPABASE_HOST, SUPABASE_USER, SUPABASE_PASSWORD]):
        raise RuntimeError(
            "Missing Supabase credentials. Set SUPABASE_HOST, SUPABASE_USER, "
            "and SUPABASE_PASSWORD in your environment."
        )

    url = (
        f"postgresql+psycopg2://{SUPABASE_USER}:{SUPABASE_PASSWORD}"
        f"@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}"
    )
    return create_engine(url)


def migrate_table(table: str, local_engine, supabase_engine,
                  chunksize: int = 10000):
    print(f"  Migrating {table}...", end=" ", flush=True)

    with local_engine.connect() as conn:
        count = pd.read_sql(
            text(f"SELECT COUNT(*) FROM {table}"), conn
        ).iloc[0, 0]

    print(f"({count:,} rows)", end=" ", flush=True)

    offset = 0
    total_written = 0

    while offset < count:
        with local_engine.connect() as conn:
            chunk = pd.read_sql(
                text(f"SELECT * FROM {table} "
                     f"LIMIT {chunksize} OFFSET {offset}"),
                conn
            )

        if chunk.empty:
            break

        chunk.to_sql(
            table,
            supabase_engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000
        )

        total_written += len(chunk)
        offset        += chunksize
        print(f".", end="", flush=True)

    print(f" ✅ {total_written:,} rows written")


if __name__ == "__main__":
    print("Connecting to local database...")
    local_engine = get_local_engine()

    print("Connecting to Supabase...")
    supabase_engine = get_supabase_engine()

    # Check how many pitch rows already in Supabase
    with supabase_engine.connect() as conn:
        existing = pd.read_sql(
            text("SELECT COUNT(*) FROM pitches"), conn
        ).iloc[0, 0]
    print(f"Pitches already in Supabase: {existing:,}")

    # Resume pitches from where we left off
    print("\nResuming pitch migration...")
    with local_engine.connect() as conn:
        total = pd.read_sql(
            text("SELECT COUNT(*) FROM pitches"), conn
        ).iloc[0, 0]

    chunksize = 10000
    offset = existing
    total_written = 0

    print(f"Resuming from offset {offset:,}...")

    while offset < total:
        with local_engine.connect() as conn:
            chunk = pd.read_sql(
                text(f"SELECT * FROM pitches "
                     f"LIMIT {chunksize} OFFSET {offset}"),
                conn
            )

        if chunk.empty:
            break

        chunk.to_sql(
            "pitches",
            supabase_engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000
        )

        total_written += len(chunk)
        offset        += chunksize
        print(f"  {offset:,} / {total:,} ...", end="\r", flush=True)

    print(f"\n✅ Pitches complete — {total_written:,} additional rows written")

    # Now migrate remaining tables
    remaining_tables = [
        "linear_weights",
        "batter_stats",
        "pitcher_stats",
        "handedness_splits",
        "batter_pitcher_features",
    ]

    print("\nMigrating remaining tables...")
    for table in remaining_tables:
        try:
            migrate_table(table, local_engine, supabase_engine)
        except Exception as e:
            print(f"\n  ❌ Error migrating {table}: {e}")
            raise

    print("\n🎉 Migration complete!")