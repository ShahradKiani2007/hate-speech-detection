from pathlib import Path

import pandas as pd
from sqlalchemy import text

from database_connection import get_engine, PROJECT_ROOT

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "labeled_data.csv"

CLASS_NAMES = {0: "hate_speech", 1: "offensive_language", 2: "neither"}

SCHEMA = """
CREATE TABLE classes (
    class_id   INTEGER PRIMARY KEY,
    class_name TEXT NOT NULL UNIQUE
);

CREATE TABLE tweets (
    tweet_id        INTEGER PRIMARY KEY,
    text            TEXT NOT NULL,
    annotator_count INTEGER NOT NULL,
    class_id        INTEGER NOT NULL,
    FOREIGN KEY (class_id) REFERENCES classes (class_id)
);

CREATE TABLE annotations (
    tweet_id        INTEGER PRIMARY KEY,
    hate_votes      INTEGER NOT NULL,
    offensive_votes INTEGER NOT NULL,
    neither_votes   INTEGER NOT NULL,
    FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id)
);

CREATE INDEX idx_tweets_class ON tweets (class_id);
"""


def build_schema(engine):
    with engine.begin() as con:
        for tbl in ["annotations", "tweets", "classes"]:
            con.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip():
                con.execute(text(stmt))


def main():
    df = pd.read_csv(RAW_CSV)
    df = df.rename(columns={"Unnamed: 0": "tweet_id", "count": "annotator_count"})

    engine = get_engine()
    build_schema(engine)

    classes = pd.DataFrame(
        {"class_id": list(CLASS_NAMES), "class_name": list(CLASS_NAMES.values())}
    )
    classes.to_sql("classes", engine, if_exists="append", index=False)

    tweets = df[["tweet_id", "tweet", "annotator_count", "class"]].rename(
        columns={"tweet": "text", "class": "class_id"}
    )
    tweets.to_sql("tweets", engine, if_exists="append", index=False)

    annotations = df[["tweet_id", "hate_speech", "offensive_language", "neither"]].rename(
        columns={
            "hate_speech": "hate_votes",
            "offensive_language": "offensive_votes",
            "neither": "neither_votes",
        }
    )
    annotations.to_sql("annotations", engine, if_exists="append", index=False)

    with engine.connect() as con:
        n = con.execute(text("SELECT COUNT(*) FROM tweets")).scalar()
    print(f"imported {n} tweets into {Path(engine.url.database).name}")


if __name__ == "__main__":
    main()
