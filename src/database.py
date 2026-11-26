"""
Phase 5 — MySQL connection + insert/query functions.

Keeps all database interaction in one place. GUI/pipeline code just calls
insert_violation() and never touches SQL directly.
"""

import os
import sys
from datetime import datetime

import mysql.connector
from mysql.connector import Error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_connection():
    """Open and return a new MySQL connection using config.DB_CONFIG."""
    try:
        conn = mysql.connector.connect(**config.DB_CONFIG)
        return conn
    except Error as e:
        print(f"[DB ERROR] Could not connect to MySQL: {e}")
        return None


def insert_violation(violation_type, track_id=None, confidence=None,
                      snapshot_path=None, plate_number=None, video_source=None):
    """
    Insert one confirmed violation row. Returns the new violation_id on
    success, or None if the insert failed (e.g. DB not reachable) — the
    caller (GUI/pipeline) should keep running even if logging fails, just
    print a warning, so a DB hiccup doesn't crash a live demo.
    """
    conn = get_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO violations
                (type, timestamp, track_id, confidence, snapshot_path, plate_number, video_source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            violation_type,
            datetime.now(),
            track_id,
            confidence,
            snapshot_path,
            plate_number,
            video_source,
        )
        cursor.execute(query, values)
        conn.commit()
        new_id = cursor.lastrowid
        cursor.close()
        return new_id
    except Error as e:
        print(f"[DB ERROR] Insert failed: {e}")
        return None
    finally:
        conn.close()


def get_all_violations(limit=500, since=None):
    """
    Fetch violations, most recent first.
    If `since` (a datetime) is given, only violations logged at or after
    that time are returned — used to show only THIS session's violations
    in the analytics view, separate from historical ones.
    """
    conn = get_connection()
    if conn is None:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        if since is not None:
            cursor.execute(
                "SELECT * FROM violations WHERE timestamp >= %s ORDER BY timestamp DESC LIMIT %s",
                (since, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM violations ORDER BY timestamp DESC LIMIT %s", (limit,)
            )
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except Error as e:
        print(f"[DB ERROR] Fetch failed: {e}")
        return []
    finally:
        conn.close()


def get_violation_counts_by_type(since=None):
    """
    Returns {violation_type: count} — used for the analytics bar chart.
    If `since` is given, only counts violations from that time onward.
    """
    conn = get_connection()
    if conn is None:
        return {}

    try:
        cursor = conn.cursor()
        if since is not None:
            cursor.execute(
                "SELECT type, COUNT(*) FROM violations WHERE timestamp >= %s GROUP BY type",
                (since,),
            )
        else:
            cursor.execute("SELECT type, COUNT(*) FROM violations GROUP BY type")
        rows = cursor.fetchall()
        cursor.close()
        return {row[0]: row[1] for row in rows}
    except Error as e:
        print(f"[DB ERROR] Count query failed: {e}")
        return {}
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick manual test: python src/database.py
    test_id = insert_violation(
        violation_type="TEST",
        track_id=999,
        confidence=0.99,
        snapshot_path="data/snapshots/test.jpg",
        video_source="test_video.mp4",
    )
    if test_id:
        print(f"Test row inserted with violation_id={test_id}")
        print("All violations:", get_all_violations(limit=5))
        print("Counts by type:", get_violation_counts_by_type())
    else:
        print("Insert failed — check your MySQL connection/credentials in config.py")