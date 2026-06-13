"""
auth/database.py
SQLite database layer.
Schema: users, sessions, audit_logs, molecules, projects, analyses
Designed to migrate to PostgreSQL with zero code changes — just swap DB_URL.
"""

import sqlite3
import os
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("QMOLSIM_DB", "data/qmolsim.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables. Safe to run multiple times."""
    with db() as conn:
        conn.executescript("""
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT 'analyst',
            department  TEXT,
            full_name   TEXT,
            is_active   INTEGER DEFAULT 1,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login  TIMESTAMP,
            created_by  INTEGER
        );

        -- Sessions table (JWT alternative — server-side sessions)
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            token       TEXT UNIQUE NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at  TIMESTAMP NOT NULL,
            ip_address  TEXT,
            user_agent  TEXT,
            is_active   INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- Audit log (21 CFR Part 11 compliant)
        CREATE TABLE IF NOT EXISTS audit_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id     INTEGER,
            username    TEXT,
            action      TEXT NOT NULL,
            resource    TEXT,
            resource_id TEXT,
            details     TEXT,
            ip_address  TEXT,
            status      TEXT DEFAULT 'SUCCESS',
            session_id  TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- Molecule database
        CREATE TABLE IF NOT EXISTS molecules (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles      TEXT NOT NULL,
            name        TEXT,
            category    TEXT,
            created_by  INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes       TEXT,
            FOREIGN KEY (created_by) REFERENCES users(id)
        );

        -- Projects (group molecules and analyses)
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            description TEXT,
            owner_id    INTEGER NOT NULL,
            status      TEXT DEFAULT 'active',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(id)
        );

        -- Analysis results (persistent storage)
        CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            molecule_id     INTEGER,
            project_id      INTEGER,
            user_id         INTEGER NOT NULL,
            analysis_type   TEXT NOT NULL,
            smiles          TEXT NOT NULL,
            molecule_name   TEXT,
            result_json     TEXT NOT NULL,
            api_score       REAL,
            status          TEXT DEFAULT 'completed',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes           TEXT,
            FOREIGN KEY (molecule_id) REFERENCES molecules(id),
            FOREIGN KEY (project_id)  REFERENCES projects(id),
            FOREIGN KEY (user_id)     REFERENCES users(id)
        );

        -- Electronic signatures (21 CFR Part 11)
        CREATE TABLE IF NOT EXISTS signatures (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            analysis_id INTEGER NOT NULL,
            meaning     TEXT NOT NULL,
            signed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            signature   TEXT NOT NULL,
            ip_address  TEXT,
            FOREIGN KEY (user_id)     REFERENCES users(id),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_audit_user     ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_time     ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_action   ON audit_logs(action);
        CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
        CREATE INDEX IF NOT EXISTS idx_analyses_user  ON analyses(user_id);
        CREATE INDEX IF NOT EXISTS idx_analyses_proj  ON analyses(project_id);
        """)

    logger.info(f"Database initialised: {DB_PATH}")
    _create_default_admin()


def _create_default_admin():
    """Create default admin user if no users exist."""
    from auth.auth import hash_password
    with db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            conn.execute("""
                INSERT INTO users (username, email, password_hash, role,
                                   department, full_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "admin",
                "admin@msn-labs.com",
                hash_password("Admin@123"),
                "admin",
                "IT",
                "System Administrator",
            ))
            logger.info("Default admin created: admin / Admin@123")
            logger.info("CHANGE THIS PASSWORD IMMEDIATELY IN PRODUCTION")
