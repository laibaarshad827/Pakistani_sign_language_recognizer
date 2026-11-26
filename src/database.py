
import sqlite3
from pathlib import Path
from contextlib import contextmanager


class Database:
    def __init__(self, db_path="data/psl_trainer.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    current_level TEXT DEFAULT 'Beginner',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sign_progress (
                    user_id INTEGER NOT NULL,
                    sign TEXT NOT NULL,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0,
                    last_attempt_at TEXT,
                    PRIMARY KEY (user_id, sign),
                    FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS learning_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    sign TEXT NOT NULL,
                    success INTEGER NOT NULL CHECK(success IN (0, 1)),
                    predicted_sign TEXT,
                    confidence REAL,
                    feedback TEXT,
                    attempted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS q_values (
                    user_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    q_value REAL DEFAULT 0.0,
                    PRIMARY KEY (user_id, state, action),
                    FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_attempts_user
                ON learning_attempts(user_id);

                CREATE INDEX IF NOT EXISTS idx_attempts_date
                ON learning_attempts(user_id, attempted_at);
            """)

    def execute(self, query, parameters=()):
        with self.connect() as conn:
            cursor = conn.execute(query, parameters)
            return cursor.lastrowid

    def fetch_one(self, query, parameters=()):
        with self.connect() as conn:
            return conn.execute(query, parameters).fetchone()

    def fetch_all(self, query, parameters=()):
        with self.connect() as conn:
            return conn.execute(query, parameters).fetchall()
