# Hate Speech Detection

A data pipeline that turns raw tweets into a model-ready dataset for classifying text as
**hate speech**, **offensive language**, or **neither**. It is built on the
[Hate Speech and Offensive Language dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
(Davidson et al., ~25k labelled tweets).

The repository covers the data-engineering side of the problem: loading the tweets into a
database, exploratory analysis, text cleaning, and feature engineering into a sparse
feature matrix that a classifier can consume directly.

## Pipeline

`python pipeline.py` runs four steps in order:

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

## Setup

```
pip install -r requirements.txt
python pipeline.py
```

Python 3.11+. Storage is local SQLite, so no database server is required.

## Analysis

```
python eda/eda.py          # figures -> eda/figures/
python sql/run_queries.py  # query results -> sql/query_outputs.txt
```

The database schema is documented in `docs/SCHEMA.md` and the EDA write-up in
`docs/EDA_REPORT.md`.

## CI

Every push and pull request to `main` runs the full pipeline and the SQL queries on
Python 3.12 via GitHub Actions (`.github/workflows/ci.yml`).

## Docker

```
docker build -t hate-speech-detection .
docker run --rm hate-speech-detection
```

See `k8s/README.md` to run the pipeline as a Job on Minikube.
