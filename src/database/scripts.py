from pathlib import Path
from .db import Base
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine
from loguru import logger


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
