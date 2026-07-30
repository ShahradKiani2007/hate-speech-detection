import pickle
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from data_splits import load_train_augmented, load_split
from database_connection import PROJECT_ROOT

SEED = 42
PROC = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

CORR_THRESHOLD = 0.95
TFIDF_MAX_FEATURES = 5000

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
ELONG_RE = re.compile(r"(.)\1{2,}")          # same char 3+ times in a row
ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")


def structural_features(text):
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    alpha = [c for c in text if c.isalpha()]
    f = {
        "char_count": n_chars,
        "word_count": n_words,
        "avg_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        "unique_word_ratio": len(set(w.lower() for w in words)) / n_words if n_words else 0.0,
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
        "hashtag_count": len(HASHTAG_RE.findall(text)),
        "mention_count": len(MENTION_RE.findall(text)),
        "url_count": len(URL_RE.findall(text)),
        "digit_count": sum(c.isdigit() for c in text),
        "allcaps_word_count": len(ALLCAPS_RE.findall(text)),
        "uppercase_ratio": (sum(c.isupper() for c in alpha) / len(alpha)) if alpha else 0.0,
        "elongated_count": len(ELONG_RE.findall(text)),
    }
    return f


def drop_correlated(df, threshold):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return df.drop(columns=to_drop), to_drop, corr


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    # train = the real train split + the augmented rows; validation stays clean.
    train_df = load_train_augmented()
    val_df = load_split("validation")

    train_feats = pd.DataFrame(
        [structural_features(t) for t in train_df["text"]],
        index=train_df.index
    )

    val_feats = pd.DataFrame(
        [structural_features(t) for t in val_df["text"]],
        index=val_df.index
    )


    train_feats, dropped, corr = drop_correlated(
        train_feats,
        CORR_THRESHOLD
    )

    val_feats = val_feats.drop(columns=dropped)

    corr.to_csv(RESULTS / "structural_correlation.csv")
    print(f"structural features: {train_feats.shape[1]} kept, dropped {dropped} (corr > {CORR_THRESHOLD})")

    scaler = StandardScaler()

    train_struct_scaled = scaler.fit_transform(train_feats)
    val_struct_scaled = scaler.transform(val_feats)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(
        train_df["clean_text"]
    )

    val_tfidf = vectorizer.transform(
        val_df["clean_text"]
    )

    print(f"train_tfidf matrix: {train_tfidf.shape}")
    print(f"val_tfidf matrix: {val_tfidf.shape}")

    X_train = sparse.hstack(
        [
            sparse.csr_matrix(train_struct_scaled),
            train_tfidf
        ],
        format="csr"
    )

    X_val = sparse.hstack(
        [
            sparse.csr_matrix(val_struct_scaled),
            val_tfidf
        ],
        format="csr"
    )

    feature_names = list(train_feats.columns) + [
        f"tfidf__{w}" for w in vectorizer.get_feature_names_out()
    ]

    train_structural_out = pd.concat(
        [
            train_df[["tweet_id", "class_id", "class_name"]].reset_index(drop=True),
            train_feats.reset_index(drop=True)
        ],
        axis=1,
    )

    val_structural_out = pd.concat(
        [
            val_df[["tweet_id", "class_id", "class_name"]].reset_index(drop=True),
            val_feats.reset_index(drop=True)
        ],
        axis=1,
    )



    train_structural_out.to_csv(PROC / "features_structural_train.csv", index=False)
    val_structural_out.to_csv(PROC / "features_structural_val.csv", index=False)

    sparse.save_npz(PROC / "X_train.npz", X_train)
    sparse.save_npz(PROC / "X_val.npz", X_val)


    np.save(PROC / "y_train.npy", train_df["class_id"].to_numpy())
    np.save(PROC / "y_val.npy", val_df["class_id"].to_numpy())

    with open(PROC / "feature_names.pkl", "wb") as fh:
        pickle.dump(feature_names, fh)
    with open(PROC / "tfidf_vectorizer.pkl", "wb") as fh:
        pickle.dump(vectorizer, fh)
    with open(PROC / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    sparse.save_npz(PROC / "tfidf_train.npz", train_tfidf)
    sparse.save_npz(PROC / "tfidf_val.npz", val_tfidf)

    print(f"saved artifacts to {PROC}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
