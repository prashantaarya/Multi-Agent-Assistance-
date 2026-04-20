# core/database.py
"""
Database Connection Layer

Currently: SQLite (local, zero setup)
Future: PostgreSQL (uncomment when ready for production)

Usage:
    from core.database import get_db, init_db
    
    # Initialize on startup
    await init_db()
    
    # Use in code
    db = get_db()
    db.execute("SELECT * FROM contacts WHERE email = ?", (email,))
"""

import os
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Local SQLite path
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/jarvis.db")

# ══════════════════════════════════════════════════════════════════════════════
# FUTURE: PostgreSQL Configuration (uncomment when ready)
# ══════════════════════════════════════════════════════════════════════════════
# import asyncpg
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.orm import sessionmaker
#
# POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/jarvis")
# engine = create_async_engine(POSTGRES_URL, echo=False)
# AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
#
# async def get_postgres_session():
#     async with AsyncSessionLocal() as session:
#         yield session
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# SQLite Implementation (Current - Local)
# ──────────────────────────────────────────────────────────────────────────────

_connection: Optional[sqlite3.Connection] = None


def _ensure_data_dir():
    """Ensure the data directory exists"""
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """
    Initialize database and create tables.
    Call this once on application startup.
    """
    global _connection
    _ensure_data_dir()
    
    _connection = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    _connection.row_factory = sqlite3.Row  # Return rows as dicts
    
    # Enable foreign keys
    _connection.execute("PRAGMA foreign_keys = ON")
    
    # Create tables
    _create_tables(_connection)
    
    logger.info(f"Database initialized: {SQLITE_PATH}")


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist"""
    
    # Contacts table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            phone TEXT,
            relationship TEXT DEFAULT 'unknown',
            organization TEXT,
            notes TEXT,
            interaction_count INTEGER DEFAULT 0,
            last_interaction TEXT,
            sentiment TEXT DEFAULT 'neutral',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index on email for fast lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email)
    """)
    
    # Preferences table (key-value store for learned preferences)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            source TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, key)
        )
    """)
    
    # Episodic memory (significant events)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            details TEXT,
            participants TEXT,
            importance TEXT DEFAULT 'normal',
            occurred_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Action history (audit log)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS action_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            capability TEXT,
            input_summary TEXT,
            output_summary TEXT,
            status TEXT DEFAULT 'completed',
            request_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Pending approvals queue
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_approvals (
            id TEXT PRIMARY KEY,
            action_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            preview TEXT,
            risk_level TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            resolved_at TEXT
        )
    """)
    
    conn.commit()
    logger.info("Database tables created/verified")


def get_db() -> sqlite3.Connection:
    """
    Get database connection.
    Initializes if not already done.
    """
    global _connection
    if _connection is None:
        init_db()
    return _connection


@contextmanager
def get_cursor():
    """
    Context manager for database cursor.
    Auto-commits on success, rolls back on error.
    
    Usage:
        with get_cursor() as cursor:
            cursor.execute("INSERT INTO contacts ...")
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise


def execute(query: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute a query and return cursor"""
    return get_db().execute(query, params)


def fetch_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Execute query and fetch one row as dict"""
    cursor = execute(query, params)
    row = cursor.fetchone()
    return dict(row) if row else None


def fetch_all(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute query and fetch all rows as list of dicts"""
    cursor = execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def close_db() -> None:
    """Close database connection (call on shutdown)"""
    global _connection
    if _connection:
        _connection.close()
        _connection = None
        logger.info("Database connection closed")


# ══════════════════════════════════════════════════════════════════════════════
# FUTURE: PostgreSQL Implementation (uncomment when ready)
# ══════════════════════════════════════════════════════════════════════════════
#
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.sql import func
#
# Base = declarative_base()
#
# class ContactModel(Base):
#     __tablename__ = "contacts"
#     
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(255), nullable=False)
#     email = Column(String(255), unique=True, index=True)
#     phone = Column(String(50))
#     relationship = Column(String(50), default="unknown")
#     organization = Column(String(255))
#     notes = Column(Text)
#     interaction_count = Column(Integer, default=0)
#     last_interaction = Column(DateTime)
#     sentiment = Column(String(20), default="neutral")
#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
#
# class PreferenceModel(Base):
#     __tablename__ = "preferences"
#     
#     id = Column(Integer, primary_key=True, index=True)
#     category = Column(String(100), nullable=False)
#     key = Column(String(100), nullable=False)
#     value = Column(Text, nullable=False)
#     confidence = Column(Float, default=0.5)
#     source = Column(String(255))
#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
#
# async def init_postgres():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     logger.info("PostgreSQL tables created")
# ══════════════════════════════════════════════════════════════════════════════
