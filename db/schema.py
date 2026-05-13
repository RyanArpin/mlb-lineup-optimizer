# db/schema.py
"""
Applies schema.sql to the database.
Run this any time you want to reset and rebuild all tables.
WARNING: DROP TABLE CASCADE will delete all existing data.
"""

import os
from db.connection import get_connection


def apply_schema():
    # Build path to schema.sql relative to this file's location
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    with open(schema_path, "r") as f:
        sql = f.read()

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        print("✅ Schema applied successfully.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Schema error: {e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    apply_schema()