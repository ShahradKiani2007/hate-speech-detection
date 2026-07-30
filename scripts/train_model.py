import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from lstm_nn import TweetDataset, LSTMClassifier, train_epoch, eval_epoch, encode, build_vocab
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import pandas as pd
import torch
from data_splits import load_train_augmented, load_split
from database_connection import PROJECT_ROOT

SEED = 42
PROC = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"
CLASS_NAMES = ["hate", "offensive", "neither"]

EXPERIMENT = "hate-speech-detection"
REGISTERED_MODEL = "hate-speech-classifier"

EPOCHS = 30
PATIENCE = 3


# each candidate is a fresh estimator plus a small grid to tune over with CV.
# the grids double as our regularization knobs (C for the linear models,
# tree depth / leaf size for the ensembles).
CANDIDATES = {
    "logreg": (
        LogisticRegression(
            solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        {"C": [0.3, 1.0, 3.0]},
    ),
    "linear_svc": (
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED),
        {"C": [0.3, 1.0, 3.0]},
    ),
    "random_forest": (
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
        ),
        {"max_depth": [None, 40], "min_samples_leaf": [1, 2]},
    ),
    "lightgbm": (
        LGBMClassifier(
            n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1
        ),
        {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1]},
    ),
}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_ml_classic():
    X_train = sparse.load_npz(PROC / "X_train.npz")
    X_val = sparse.load_npz(PROC / "X_val.npz")

    y_train = np.load(PROC / "y_train.npy")
    y_val = np.load(PROC / "y_val.npy")
    comparison = []
    best = {"f1_macro": -1}

    for name, (estimator, grid) in CANDIDATES.items():
        print("\n=== tuning {} ===".format(name))
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=3, n_jobs=-1)
        search.fit(X_train, y_train)

        model = search.best_estimator_
        val_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, val_pred)
        print("  best params: {}".format(search.best_params_))
        print("  val f1_macro: {:.4f}  accuracy: {:.4f}".format(
            metrics["f1_macro"], metrics["accuracy"]))

        cm_path = RESULTS / "confusion_val_{}.png".format(name)
        plot_confusion(y_val, val_pred, "{} (validation)".format(name), cm_path)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_f1_macro", search.best_score_)
            mlflow.log_metrics({"val_" + k: v for k, v in metrics.items()})
            mlflow.log_artifact(str(cm_path))
            info = mlflow.sklearn.log_model(
                model, name="model", serialization_format="cloudpickle"
            )

        row = {"model": name, "best_params": search.best_params_,
               "cv_f1_macro": search.best_score_}
        row.update({"val_" + k: v for k, v in metrics.items()})
        comparison.append(row)

        if metrics["f1_macro"] > best["f1_macro"]:
            best = {"name": name, "f1_macro": metrics["f1_macro"],
                    "estimator": estimator, "params": search.best_params_,
                    "model_uri": info.model_uri}

    import pandas as pd
    pd.DataFrame(comparison).to_csv(RESULTS / "model_comparison.csv", index=False)

    print("\nbest model on validation: {} (f1_macro={:.4f})".format(
        best["name"], best["f1_macro"]))

    # refit the winner on train + validation so the final model sees more data,
    # then keep the test split untouched for honest evaluation later.
    X_train_val = sparse.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    final = best["estimator"].set_params(**best["params"])
    model_label = best["name"]
    # LinearSVC has no predict_proba; calibrate so we can store class probabilities
    # in the database and report probability-based metrics (AUC-ROC, log loss).
    if not hasattr(final, "predict_proba"):
        final = CalibratedClassifierCV(final, cv=3)
        model_label = best["name"] + " (calibrated)"
    final.fit(X_train_val, y_train_val)

    with open(MODELS / "model.pkl", "wb") as fh:
        pickle.dump(final, fh)

    meta = {
        "model": model_label,
        "params": best["params"],
        "val_f1_macro": best["f1_macro"],
        "n_features": X_train.shape[1],
        "class_names": CLASS_NAMES,
        "seed": SEED,
    }
    with open(MODELS / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    mlflow.register_model(best["model_uri"], REGISTERED_MODEL)
    print("saved final model -> models/model.pkl and registered as '{}'".format(
        REGISTERED_MODEL))



def train_lstm():
    print("== training lstm ==")
    # train = real train split + augmented rows; validation untouched.
    train_df = load_train_augmented()
    val_df = load_split("validation")
    X_train_text = train_df["clean_text"].astype(str)
    X_val_text = val_df["clean_text"].astype(str)
    y_train = train_df["class_id"]
    y_val = val_df["class_id"]

    vocab = build_vocab(X_train_text, max_vocab=20000, min_freq=1)
    vocab_size = len(vocab)
    print("vocab size:", vocab_size)

    max_len = max(X_train_text.str.split().str.len())
    X_train_ids = np.array([encode(t, vocab, max_len) for t in X_train_text])
    X_val_ids = np.array([encode(t, vocab, max_len) for t in X_val_text])
    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    sentences = [t.split() for t in X_train_text]
    print("== running Word2Vec ==")
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,    
        workers=4,
        sg=1,
        epochs=20
    )
    w2v_model.save(str(MODELS / "w2v_hatespeech.model"))

    embedding_dim = 100
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    hits, misses = 0, 0
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            hits += 1
        else:
            misses += 1  # <PAD>, <OOV>, and truly unseen words stay zero
    np.save(MODELS / "embedding_matrix.npy", embedding_matrix)
    print(f"hits: {hits}, misses: {misses}")
    train_ds = TweetDataset(X_train_ids, y_train_arr)
    val_ds = TweetDataset(X_val_ids, y_val_arr)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(CLASS_NAMES)

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        embedding_matrix=embedding_matrix,
        hidden_dim=64,
        num_classes=num_classes,
        pad_idx=vocab["<PAD>"],
        bidirectional=True,
        dropout=0.3,
        freeze_embeddings=False  # True to keep w2v vectors fixed
    ).to(device)
    classes = np.unique(y_train_arr)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_arr)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    patience_counter = 0
    print("== training model ==")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

        with open(MODELS / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

        with open(MODELS / "lstm_meta.json", "w") as f:
            json.dump(
                {
                    "model" : "lstm",
                    "max_len": max_len,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": 64,
                    "bidirectional": True,
                },
                f,
                indent=2,
            )

def train_bert():
    # HateBERT fine-tuning is heavy and really wants a GPU, so it is env-tunable:
    #   BERT_SKIP=1            skip it entirely (used by CI / quick classic+lstm runs)
    #   BERT_TRAIN_FRACTION=x  subsample the training rows (0<x<=1) for a fast smoke run
    #   BERT_EPOCHS=n          number of fine-tuning epochs (default 2)
    if os.getenv("BERT_SKIP"):
        print("== BERT_SKIP set — skipping HateBERT fine-tuning ==")
        return

    from scipy.special import softmax
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
    )
    from bert_nn import MODEL_NAME, MAX_LENGTH, to_dataset, tune_hate_threshold

    print("== training hateBERT ==")
    epochs = int(os.getenv("BERT_EPOCHS", "2"))
    fraction = float(os.getenv("BERT_TRAIN_FRACTION", "1"))

    train_df = load_train_augmented()
    if fraction < 1.0:
        train_df = train_df.sample(frac=fraction, random_state=SEED).reset_index(drop=True)
        print("subsampled the training set to {:.0%} ({} rows)".format(fraction, len(train_df)))
    val_df = load_split("validation")
    for d in (train_df, val_df):
        d["clean_text"] = d["clean_text"].astype(str)
        d["class_id"] = d["class_id"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
    )

    train_dataset = to_dataset(train_df, tokenizer)
    val_dataset = to_dataset(val_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(MODELS / "hateBERT_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(RESULTS / "logs"),
        logging_steps=50,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1)
        ),
    )

    trainer.train()
    val_results = trainer.evaluate()
    print("\nvalidation results:")
    for k, v in val_results.items():
        print(" ", k, v)

    # threshold tuning belongs on validation, never on test
    val_output = trainer.predict(val_dataset)
    val_probs = softmax(val_output.predictions, axis=1)
    best_threshold, best_val_f1 = tune_hate_threshold(val_probs, val_output.label_ids)
    print("\nbest P(hate) threshold on validation: {:.2f} (macro F1 {:.4f})".format(
        best_threshold, best_val_f1))

    final_path = MODELS / "hateBERT_final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    val_metrics = {k: float(v) for k, v in val_results.items() if isinstance(v, (int, float))}
    meta = {
        "model": "hateBERT",
        "model_name": MODEL_NAME,
        "epochs": epochs,
        "max_length": MAX_LENGTH,
        "learning_rate": 2e-5,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "hate_threshold": best_threshold,
        "val_macro_f1_at_threshold": float(best_val_f1),
        "val_metrics": val_metrics,
    }
    with open(RESULTS / "hatebert_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    with mlflow.start_run(run_name="hateBERT"):
        mlflow.log_param("model", "hateBERT")
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("hate_threshold", best_threshold)
        mlflow.log_metric("val_macro_f1_at_threshold", best_val_f1)
        mlflow.log_metrics({"val_" + k.replace("eval_", ""): v for k, v in val_metrics.items()})

    print("saved fine-tuned model -> models/hateBERT_final")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    train_ml_classic()
    train_lstm()
    train_bert()




if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()
