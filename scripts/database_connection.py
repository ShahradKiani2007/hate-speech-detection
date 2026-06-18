import os
from pathlib import Path

from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("DB_PATH", PROJECT_ROOT / "database" / "dataset.db"))


def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}")


def get_connection():
    return get_engine().connect()


if __name__ == "__main__":
    with get_engine().connect() as con:
        print("connected to", DB_PATH)
