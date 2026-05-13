# sql/run_features.py
"""
Executes feature_engineering.sql against the database.
Produces the batter_pitcher_features table used for ML training.
"""

import os
import pandas as pd
from db.connection import get_connection, get_engine


def run_feature_engineering():
    """
    Reads and executes feature_engineering.sql.
    Prints row counts for each intermediate table created.
    """
    sql_path = os.path.join(os.path.dirname(__file__), "feature_engineering.sql")

    with open(sql_path, "r") as f:
        sql = f.read()

    print("Running feature engineering SQL...")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        print("✅ Feature engineering complete.\n")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        cursor.close()
        conn.close()
        return
    finally:
        cursor.close()
        conn.close()

    # Print row counts for all feature tables
    engine = get_engine()
    tables = [
        "linear_weights",
        "batter_stats",
        "pitcher_stats",
        "handedness_splits",
        "batter_pitcher_features"
    ]
    print("Table row counts:")
    print(f"  {'Table':<30} {'Rows':>10}")
    print(f"  {'-'*40}")
    for table in tables:
        count = pd.read_sql(f"SELECT COUNT(*) FROM {table}", engine).iloc[0, 0]
        print(f"  {table:<30} {count:>10,}")


def load_features() -> pd.DataFrame:
    """
    Loads the final feature table into a pandas DataFrame.
    Called by the ML model in Step 5.
    """
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM batter_pitcher_features", engine)
    print(f"✅ Loaded {len(df):,} matchup rows, {len(df.columns)} features")
    return df


if __name__ == "__main__":
    run_feature_engineering()