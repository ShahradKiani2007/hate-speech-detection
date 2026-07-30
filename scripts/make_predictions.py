import json
import os
import pickle

from llm import LlmModel
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report, log_loss, roc_auc_score
from sqlalchemy import text

from database_connection import PROJECT_ROOT, get_engine
from preprocess import clean_text
from feature_engineering import structural_features
from train_model import CLASS_NAMES, compute_metrics, plot_confusion
from lstm_nn import LSTMClassifier, encode, TweetDataset, eval_epoch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

PROC = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"

PRED_TABLE = "predictions"
SCHEMA ="""
    CREATE TABLE IF NOT EXISTS predictions (
        tweet_id       INTEGER,
        true_class_id  INTEGER,
        pred_class_id  INTEGER NOT NULL,
        pred_class     TEXT NOT NULL,
        prob_hate      REAL,
        prob_offensive REAL,
        prob_neither   REAL,
        model_name     TEXT NOT NULL,
        PRIMARY KEY (tweet_id, model_name),
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
    model_name = rows["model_name"].iloc[0]
    with engine.begin() as con:
        con.execute(text(SCHEMA))
        con.execute(
            text("DELETE FROM predictions WHERE model_name = :model_name"),
            {"model_name": model_name},
        )
    rows.to_sql(PRED_TABLE, engine, if_exists="append", index=False)
    with engine.connect() as con:
            n = con.execute(
                text("SELECT COUNT(*) FROM predictions WHERE model_name = :model_name"),
                {"model_name": model_name},
            ).scalar()
    print("wrote {} rows for model '{}' to '{}' table".format(n, model_name, PRED_TABLE))


def evaluation(y_true, y_pred, proba, meta, df):
    model_name = meta["model"]

    metrics = compute_metrics(y_true, y_pred)

    if proba is not None:
        metrics["auc_roc_ovr"] = roc_auc_score(
            y_true,
            proba,
            multi_class="ovr",
            average="macro"
        )
        metrics["log_loss"] = log_loss(
            y_true,
            proba,
            labels=[0, 1, 2]
        )

    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True
    )

    print("\ntest metrics ({} model):".format(model_name))

    for k, v in metrics.items():
        print("  {:18s} {:.4f}".format(k, v))

    print("\n" + classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    ))

    with open(RESULTS / f"test_metrics_{model_name}.json", "w") as fh:
        json.dump(
            {
                "model": model_name,
                "metrics": metrics,
                "classification_report": cls_report
            },
            fh,
            indent=2
        )

    plot_confusion(
        y_true,
        y_pred,
        "{} (test)".format(model_name),
        RESULTS / f"confusion_test_{model_name}.png"
    )

    rows = pd.DataFrame({
        "tweet_id": df["tweet_id"].to_numpy(),
        "true_class_id": y_true,
        "pred_class_id": y_pred,
        "pred_class": [CLASS_NAMES[p] for p in y_pred],
        "prob_hate": proba[:, 0] if proba is not None else np.nan,
        "prob_offensive": proba[:, 1] if proba is not None else np.nan,
        "prob_neither": proba[:, 2] if proba is not None else np.nan,
        "model_name": model_name,
    })

    save_predictions(rows)

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    # the held-out test split, written by split_data.py — the same rows every model
    # is scored on. LLM / BERT can be skipped via env when the API key / GPU is absent.
    df = pd.read_csv(PROC / "test.csv")
    df["text"] = df["text"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)

    classic_ml_model(df)
    if os.getenv("LLM_SKIP"):
        print("== LLM_SKIP set — skipping the LLM model ==")
    else:
        llm_model(df)
    lstm_model(df)
    if os.getenv("BERT_SKIP"):
        print("== BERT_SKIP set — skipping the hateBERT model ==")
    else:
        bert_model(df)

def classic_ml_model(df):
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

    df["clean_text"] = df["text"].apply(clean_text)

    X_test = build_features(df, scaler, vectorizer, struct_cols)
    y_true = df["class_id"].to_numpy()

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    evaluation(y_true, y_pred, proba, meta, df)

def llm_model(df):
    y_true = df["class_id"].to_numpy()
    model = LlmModel(batch_size=5, max_concurrency=5, group_size=10)
    y_pred = model.predict(df["text"].tolist(), checkpoint_path="checkpoints/llm_predictions.json")
    meta = {'model' : 'llm'}
    evaluation(y_true, y_pred, None, meta, df)

def lstm_model(df):
    df["clean_text"] = df["text"].apply(clean_text)
    X_test_text = df["clean_text"]
    y_true = df["class_id"].to_numpy()
    with open(MODELS / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open(MODELS / "lstm_meta.json") as f:
        lstm_meta = json.load(f)

    embedding_dim = lstm_meta["embedding_dim"]
    hidden_dim = lstm_meta["hidden_dim"]
    bidirectional = lstm_meta["bidirectional"]
    max_len = lstm_meta["max_len"]
    embedding_matrix = np.load(MODELS / "embedding_matrix.npy")
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        num_classes=3,
        pad_idx=vocab["<PAD>"],
        bidirectional=bidirectional,
        dropout=0.3,
        freeze_embeddings=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(MODELS / "best_model.pt", map_location=device)
    )

    model.to(device)
    model.eval()

    X_test_ids = np.array([
        encode(t, vocab, max_len)
        for t in X_test_text
    ])
    test_ds = TweetDataset(X_test_ids, y_true)
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False
    )
    
    model.eval()

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    proba = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    evaluation(y_true, y_pred, proba, lstm_meta, df)

def bert_model(df):
    from scipy.special import softmax
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from bert_nn import MAX_LENGTH

    final_path = MODELS / "hateBERT_final"
    tokenizer = AutoTokenizer.from_pretrained(final_path)
    model = AutoModelForSequenceClassification.from_pretrained(final_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df["clean_text"].astype(str).tolist()
    y_true = df["class_id"].to_numpy()

    all_logits = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i:i + batch_size],
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.cpu().numpy()
            all_logits.append(logits)

    proba = softmax(np.concatenate(all_logits), axis=1)
    y_pred = proba.argmax(axis=1)

    with open(RESULTS / "hatebert_meta.json") as fh:
        meta = json.load(fh)

    evaluation(y_true, y_pred, proba, meta, df)

if __name__ == "__main__":
    main()
