import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn

from database_connection import PROJECT_ROOT

SEED = 42
PROC = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
MODELS = PROJECT_ROOT / "models"
CLASS_NAMES = ["hate", "offensive", "neither"]

EXPERIMENT = "hate-speech-detection"
REGISTERED_MODEL = "hate-speech-classifier"

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


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    X_train = sparse.load_npz(PROC / "X_train.npz")
    X_val = sparse.load_npz(PROC / "X_val.npz")

    y_train = np.load(PROC / "y_train.npy")
    y_val = np.load(PROC / "y_val.npy")

    mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

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


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
