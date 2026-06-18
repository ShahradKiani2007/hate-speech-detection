import pandas as pd

from database_connection import get_engine, PROJECT_ROOT

QUERY = """
SELECT t.tweet_id,
       t.text,
       t.annotator_count,
       t.class_id,
       c.class_name,
       a.hate_votes,
       a.offensive_votes,
       a.neither_votes
FROM tweets t
JOIN classes c ON t.class_id = c.class_id
JOIN annotations a ON t.tweet_id = a.tweet_id
ORDER BY t.tweet_id
"""

OUT = PROJECT_ROOT / "data" / "processed" / "loaded.pkl"


def load_data():
    engine = get_engine()
    return pd.read_sql(QUERY, engine)


def main():
    df = load_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUT)
    print(f"loaded {len(df)} rows from database -> {OUT.name}")


if __name__ == "__main__":
    main()
