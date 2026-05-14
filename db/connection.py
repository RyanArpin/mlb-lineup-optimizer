# db/connection.py
"""
Database connection utilities for the MLB Lineup Optimizer.

When running on Streamlit Cloud: reads credentials from st.secrets
When running locally: reads credentials from .env file
"""

import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


def _get_db_params():
    """
    Returns database connection parameters.
    Uses Streamlit secrets when deployed, .env when local.
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            db = st.secrets["database"]
            return {
                "host":     db["host"],
                "port":     int(db["port"]),
                "dbname":   db["name"],
                "user":     db["user"],
                "password": db["password"],
            }
    except Exception:
        pass

    # Fall back to .env
    return {
        "host":     os.getenv("DB_HOST"),
        "port":     int(os.getenv("DB_PORT", 5432)),
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def get_connection():
    """
    Returns a raw psycopg2 connection.
    Use for direct SQL execution and DDL statements.
    """
    return psycopg2.connect(**_get_db_params())


def get_engine():
    """
    Returns a SQLAlchemy engine.
    Use with pandas .read_sql() and .to_sql().
    """
    p = _get_db_params()
    url = (
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['dbname']}"
    )
    return create_engine(url)


def get_conn_for_pandas():
    """
    Returns a SQLAlchemy connection for pandas compatibility.
    """
    engine = get_engine()
    return engine.connect()


def test_connection():
    """
    Sanity check — confirms database connection works.
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


if __name__ == "__main__":
    test_connection()