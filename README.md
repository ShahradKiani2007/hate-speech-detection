# Hate Speech Detection

A data pipeline that turns raw tweets into a model-ready dataset for classifying text as
**hate speech**, **offensive language**, or **neither**. It is built on the
[Hate Speech and Offensive Language dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
(Davidson et al., ~25k labelled tweets).

The repository goes end to end: loading the tweets into a database, exploratory analysis,
text cleaning, feature engineering into a sparse feature matrix, and finally training,
evaluating, and deploying a classifier whose predictions are written back to the database.

## Data preparation

`python pipeline.py` runs the four data-prep steps in order:

1. **import_to_db** — builds the SQLite schema (`classes`, `tweets`, `annotations`) and
   loads the raw CSV.
2. **load_data** — joins the tables back into a DataFrame.
3. **preprocess** — cleans the tweet text (unescape HTML, strip URLs/mentions/RT,
   normalize) and drops tweets that become empty.
4. **feature_engineering** — builds 13 structural/lexical features plus TF-IDF (1–2 grams,
   5k terms), combines them into a sparse matrix `X` with labels `y`, and saves the fitted
   vectorizer and scaler.

Results land in `data/processed/` (`X.npz`, `y.npy`, `tfidf_vectorizer.pkl`,
`scaler.pkl`, …). The annotator vote columns are intentionally excluded from the features,
since the label is their argmax and using them would leak the target.

## Modelling and the two pipelines

Phase 3 adds the model and splits the workflow into a **training pipeline** and a
**prediction pipeline**, so training and inference are automated separately.

```
python run_train.py     # data prep -> split -> tune & compare models -> save best
python run_predict.py    # load test tweets -> transform -> predict -> write to DB
```

`run_train.py` runs the data-prep steps above, then:

5. **split_data** — stratified 70/15/15 train/val/test split, saved to
   `data/processed/splits.npz`.
6. **train_model** — tunes Logistic Regression, Linear SVM, Random Forest, and LightGBM with
   cross-validated grid search, compares them on the validation split (macro-F1), logs every
   run to MLflow, refits the winner on train+val, and saves it to `models/model.pkl`.

`run_predict.py` runs `make_predictions.py`, a self-contained prediction pipeline that
loads the held-out test tweets from the database, re-applies the **saved** preprocessing and
feature transformers (no refit — the same scaler and TF-IDF vectoriser from training),
loads the trained model, evaluates it, and writes the predictions and class probabilities
back into a `predictions` table in the database.

The Linear SVM wins (test accuracy 0.90, macro-F1 0.69, AUC-ROC 0.93). Full methodology,
the model comparison, and a business-oriented reading of the results are in
[`docs/MODEL_REPORT.md`](docs/MODEL_REPORT.md).

## Orchestration (Prefect)

The same two pipelines are also defined as Prefect flows for proper orchestration:

```
python flows.py train      # train_flow
python flows.py predict    # predict_flow
```

Each stage is a Prefect task, so you get a dependency graph, retries, and run history in the
Prefect UI (`prefect server start`).

## Experiment tracking (MLflow)

Every model run — parameters, metrics, and confusion-matrix artifacts — is logged to a
local MLflow store (`mlflow.db`), and the best model is registered as
`hate-speech-classifier`. Browse it with:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Setup

```
pip install -r requirements.txt
python run_train.py      # trains and saves the model
python run_predict.py    # scores the test set and writes predictions to the DB
```

Python 3.11+. Storage is local SQLite, so no database server is required. (`pipeline.py`
still exists if you only want the Phase 2 data-prep steps.)

## Analysis

```
python eda/eda.py          # figures -> eda/figures/
python sql/run_queries.py  # query results -> sql/query_outputs.txt
```

The database schema is documented in `docs/SCHEMA.md` and the EDA write-up in
`docs/EDA_REPORT.md`.

## CI

Every push and pull request to `main` runs the full training and prediction pipelines and
the SQL queries on Python 3.12 via GitHub Actions (`.github/workflows/ci.yml`).

## Presentation

The Phase 1–3 presentation video and slides are linked in
[`video_link.txt`](video_link.txt).
