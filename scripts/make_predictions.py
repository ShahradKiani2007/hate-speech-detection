import json
import pickle

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sqlalchemy import text

from database_connection import PROJECT_ROOT, get_engine
from load_data import load_data
from preprocess import clean_text
from feature_engineering import structural_features
from train_model import CLASS_NAMES, compute_metrics, plot_confusion

PROC = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"

PRED_TABLE = "predictions"
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    tweet_id       INTEGER PRIMARY KEY,
    true_class_id  INTEGER,
    pred_class_id  INTEGER NOT NULL,
    pred_class     TEXT NOT NULL,
    prob_hate      REAL,
    prob_offensive REAL,
    prob_neither   REAL,
    model_name     TEXT NOT NULL,
    FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id)
)
"""


def build_features(df, scaler, vectorizer, struct_cols):
    # rebuild exactly the phase-2 feature space, but with the *saved* transformers:
    # structural features come from the raw text, tf-idf from the cleaned text.
    feats = pd.DataFrame([structural_features(t) for t in df["text"]])[struct_cols]
    struct_scaled = scaler.transform(feats)
    tfidf = vectorizer.transform(df["clean_text"])
    return sparse.hstack([sparse.csr_matrix(struct_scaled), tfidf], format="csr")


def save_predictions(rows):
    engine = get_engine()
    with engine.begin() as con:
        con.execute(text(SCHEMA))
        con.execute(text("DELETE FROM predictions"))
    rows.to_sql(PRED_TABLE, engine, if_exists="append", index=False)
    with engine.connect() as con:
        n = con.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
    print("wrote {} rows to '{}' table".format(n, PRED_TABLE))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    # these pickles are our own artifacts, written by train_model.py / feature_engineering.py
    with open(MODELS / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    with open(MODELS / "metadata.json") as fh:
        meta = json.load(fh)
    with open(PROC / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(PROC / "tfidf_vectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)
    with open(PROC / "feature_names.pkl", "rb") as fh:
        feature_names = pickle.load(fh)
    struct_cols = [f for f in feature_names if not f.startswith("tfidf__")]

    splits = np.load(PROC / "splits.npz")
    test_ids = splits["tweet_ids"][splits["test_idx"]]

    # "load new data from the database": pull the held-out test tweets
    df = load_data()
    df = df[df["tweet_id"].isin(test_ids)].reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)

    X_test = build_features(df, scaler, vectorizer, struct_cols)
    y_true = df["class_id"].to_numpy()

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_true, y_pred)
    if proba is not None:
        metrics["auc_roc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        metrics["log_loss"] = log_loss(y_true, proba, labels=[0, 1, 2])

    print("\ntest metrics ({} model):".format(meta["model"]))
    for k, v in metrics.items():
        print("  {:18s} {:.4f}".format(k, v))
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    with open(RESULTS / "test_metrics.json", "w") as fh:
        json.dump({"model": meta["model"], "metrics": metrics}, fh, indent=2)
    plot_confusion(y_true, y_pred, "{} (test)".format(meta["model"]),
                   RESULTS / "confusion_test.png")

    rows = pd.DataFrame({
        "tweet_id": df["tweet_id"].to_numpy(),
        "true_class_id": y_true,
        "pred_class_id": y_pred,
        "pred_class": [CLASS_NAMES[p] for p in y_pred],
        "prob_hate": proba[:, 0] if proba is not None else np.nan,
        "prob_offensive": proba[:, 1] if proba is not None else np.nan,
        "prob_neither": proba[:, 2] if proba is not None else np.nan,
        "model_name": meta["model"],
    })
    save_predictions(rows)


if __name__ == "__main__":
    main()
