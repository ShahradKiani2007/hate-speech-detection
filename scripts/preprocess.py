import html
import re

import pandas as pd

from database_connection import PROJECT_ROOT
from load_data import load_data

OUT = PROJECT_ROOT / "data" / "processed" / "clean.pkl"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"\brt\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(t):
    t = html.unescape(t)
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r"\1", t)          # keep the word, drop the '#'
    t = RT_RE.sub(" ", t)
    t = NON_ALPHA_RE.sub(" ", t)          # drop digits, punctuation, emojis
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def main():
    df = load_data()

    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)

    df["clean_word_count"] = df["clean_text"].str.split().str.len()

    df.to_pickle(OUT)
    print(f"preprocessed {len(df)} rows ({dropped} empty after cleaning dropped) -> {OUT.name}")


if __name__ == "__main__":
    main()
