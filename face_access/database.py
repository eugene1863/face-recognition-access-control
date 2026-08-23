"""SQLite persistence for enrolled users and the access log."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT NOT NULL,
    status TEXT NOT NULL,          -- GRANTED or DENIED
    confidence REAL,
    timestamp TEXT NOT NULL,
    snapshot_path TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def add_user(name: str) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def get_or_create_user(name: str) -> int:
    existing = get_user_by_name(name)
    if existing:
        return existing["id"]
    return add_user(name)


def get_user_by_name(name: str):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


def list_users():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def log_access(user_id, name, status, confidence, snapshot_path=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO access_log (user_id, name, status, confidence, timestamp, snapshot_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                name,
                status,
                confidence,
                datetime.now().isoformat(timespec="seconds"),
                snapshot_path,
            ),
        )


def recent_log(limit=200):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM access_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
