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

Phase 3 adds the models and splits the workflow into a **training pipeline** and a
**prediction pipeline**, so training and inference are automated separately. Four model
families are trained and then all scored on the same held-out test split:

- **Classic ML** — Logistic Regression, Linear SVM, Random Forest, LightGBM over the
  structural + TF-IDF features; the best on validation (macro-F1) is refit and saved.
- **LSTM** — a bidirectional LSTM over Word2Vec embeddings trained on the corpus.
- **LLM** — a Gemini few-shot classifier (`scripts/llm.py`).
- **HateBERT** — `GroNLP/hateBERT` fine-tuned on the 3-class task.

```
python run.py            # (alias for run_train.py) train every model
python run_train.py      # data prep -> split -> augment -> features -> train all models
python run_predict.py    # load test tweets -> transform -> predict with every model -> DB
```

`run_train.py` / `run.py` runs the data-prep steps above, then:

5. **split_data** — stratified 70/15/15 train/val/test split, written as
   `train`/`validation`/`test` `.csv` + `.pkl` in `data/processed/`.
6. **make_augmentations** — LLM (Gemini) augmentation of the **hate** class:
   counterfactual / misspelled / paraphrased / quoted variants. These are added to the
   **training set only** — never validation or test. If `augmented.csv` already exists it is
   reused as-is (no LLM calls); otherwise it is regenerated (needs `GOOGLE_API_KEY`).
7. **feature_engineering** — fits the scaler + TF-IDF on the augmented training set.
8. **train_model** — trains the classic ML ensemble (CV grid search), the LSTM, and
   fine-tunes HateBERT; logs runs to MLflow and saves each model under `models/`.

`run_predict.py` runs `make_predictions.py`, which loads the held-out `test.csv`, re-applies
the **saved** transformers (no refit), runs every model, and writes each one's predictions
and class probabilities into the shared `predictions` table keyed by `(tweet_id, model_name)`.

HateBERT fine-tuning is heavy and wants a GPU, so it (and the LLM) are env-tunable:

| var | effect |
|-----|--------|
| `BERT_SKIP=1` | skip HateBERT (train and/or predict) |
| `BERT_TRAIN_FRACTION=0.1` | fine-tune on a fraction of the training rows (quick smoke run) |
| `BERT_EPOCHS=n` | number of fine-tuning epochs (default 2) |
| `LLM_SKIP=1` | skip the Gemini LLM at prediction time |

The Linear SVM is the strongest classic model (test accuracy 0.90, macro-F1 0.69,
AUC-ROC 0.93). Full methodology, the model comparison, and a business-oriented reading of the
results are in [`docs/MODEL_REPORT.md`](docs/MODEL_REPORT.md).

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

GitHub Actions (`.github/workflows/ci.yml`) runs the training and prediction pipelines and
the SQL queries on Python 3.12. HateBERT and the Gemini LLM need a GPU / API key, so CI sets
`BERT_SKIP` and `LLM_SKIP` and exercises the classic ML + LSTM models only; the committed
`augmented.csv` means the augmentation step runs offline.

## Presentation

The Phase 1–3 presentation video and slides are linked in
[`video_link.txt`](video_link.txt).
