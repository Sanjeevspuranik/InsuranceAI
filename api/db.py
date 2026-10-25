"""SQLAlchemy database setup shared by the API and worker."""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import PROJECT_ROOT, settings


database_url = settings.database_url
if database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

connect_args = {}
if database_url.startswith("sqlite:///"):
    db_path = database_url.removeprefix("sqlite:///")
    if db_path != ":memory:" and not db_path.startswith("file:"):
        path = Path(db_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False, "timeout": 30}

engine = create_engine(
    database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

if database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _configure_sqlite(connection, _record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Base(DeclarativeBase):
    pass


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


def initialize_database() -> None:
    from api import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def verify_database_schema() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("claims"):
        raise RuntimeError(
            "Database schema is missing. Run 'alembic upgrade head' first."
        )
    columns = {column["name"] for column in inspector.get_columns("claims")}
    if "reasoning_tokens" not in columns:
        raise RuntimeError(
            "Database schema is outdated. Run 'alembic upgrade head' first."
        )
