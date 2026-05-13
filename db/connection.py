# db/connection.py
"""
Database connection utilities for the MLB Lineup Optimizer.

Provides two connection types:
  - get_connection(): raw psycopg2 connection, for executing SQL directly
  - get_engine():     SQLAlchemy engine, for pandas .read_sql() / .to_sql()

Both read credentials from the .env file via python-dotenv.
"""

import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load variables from .env into the environment
# This must be called before any os.getenv() calls
load_dotenv()


def get_connection():
    """
    Returns a raw psycopg2 connection.

    Use this for:
      - CREATE TABLE / DROP TABLE (DDL statements)
      - INSERT / UPDATE / DELETE (DML statements)
      - Any query where you want manual transaction control

    Usage:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.commit()   # required after INSERT/UPDATE/DELETE
        conn.close()    # always close when done

    Or as a context manager (auto-closes on exit):
        with get_connection() as conn:
            ...
    """
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn


def get_engine():
    """
    Returns a SQLAlchemy engine.

    Use this for:
      - pd.read_sql("SELECT ...", engine)   → query results as a DataFrame
      - df.to_sql("table_name", engine)     → write a DataFrame to a table

    SQLAlchemy handles connection pooling automatically,
    so you don't need to manually open/close connections here.
    """
    db_url = (
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
        f"/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(db_url)
    return engine


def test_connection():
    """
    Sanity check — confirms PostgreSQL is running and credentials are correct.
    Run this after any environment change.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print("✅ Connected successfully!")
        print(f"   PostgreSQL version: {version[0]}")
        cursor.close()
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed:\n   {e}")
        print("\n── Troubleshooting checklist ──────────────────────────")
        print("  1. Is PostgreSQL running?")
        print("       sudo service postgresql status")
        print("       sudo service postgresql start")
        print("  2. Do your .env credentials match what you set in psql?")
        print("  3. Does the database exist?")
        print("       sudo -u postgres psql -l")
        print("  4. Is your virtual environment activated?")
        print("       source venv/bin/activate")

def get_conn_for_pandas():
    """
    Returns a psycopg2 connection wrapped for pandas compatibility.
    Use this instead of get_connection() when calling pd.read_sql().
    """
    from sqlalchemy import text
    engine = get_engine()
    return engine.connect()

if __name__ == "__main__":
    test_connection()