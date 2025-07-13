from pathlib import Path
from .db import Base
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine
from loguru import logger
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

_engines: dict[Path, "Engine"] = {}


def create_db(path: Path = ".database/data.db", echo: bool = False) -> None:
    """Creates :class:`src.database.db.Base` instance locally.

    Args:
        path Path: path to save the database. Defaults to ".database/data.db".
        echo: verbose parameter

    Raises:
        RuntimeError: If the file system is not writable or SQLAlchemy cannot create the DB.
    """
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}", echo=echo, future=True)

    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database has been initialized --> {db_path}")
    except OperationalError as exc:
        raise RuntimeError(f"Could not create database at {db_path}") from exc
    finally:
        engine.dispose()


def _get_engine(path: Path | str = ".database/data.db", *, echo: bool = False):
    db_path = Path(path).expanduser().resolve()
    if (cached := _engines.get(db_path)) and cached.echo == echo:
        return cached

    engine = create_engine(f"sqlite:///{db_path}", echo=echo, future=True)
    _engines[db_path] = engine
    return engine


def connect_to_db(
    path: Path = ".database/data.db",
    *,
    echo: bool = False,
    autoflush: bool = False,
    autocommit: bool = False,
    expire_on_commit: bool = False,
) -> Session:
    """Unable to establish the connection with database

    Args:
        path (Path | str, optional): Path to the database. Defaults to ".database/data.db".
        echo (bool, optional): Verbosse settings. Defaults to False.
        autoflush (bool, optional): The autoflush setting. Defaults to False.
        autocommit (bool, optional): The autocommit setting. Defaults to False.
        expire_on_commit (bool, optional): The expire on commit setting. Defaults to False.

    Returns:
        Session: session with database
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Database was not found: {path}. Please, create database first: `drone-rescue data create-db`"
        )

    engine = _get_engine(path, echo=echo)

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=autoflush,
        autocommit=autocommit,
        expire_on_commit=expire_on_commit,
    )
    return SessionLocal()
