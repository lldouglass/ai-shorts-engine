"""Database session management."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from shorts_engine.config import settings

# Create engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_session() -> Generator[Session, None, None]:
    """Get a database session (for FastAPI dependency injection)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_session_context() -> Generator[Session, None, None]:
    """Get a database session as a context manager (for use outside of FastAPI)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database connection and verify connectivity."""

    # Just verify we can connect
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
