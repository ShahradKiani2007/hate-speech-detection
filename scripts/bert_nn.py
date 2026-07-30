import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score

MODEL_NAME = "GroNLP/hateBERT"
MAX_LENGTH = 128
CLASS_NAMES = ["hate", "offensive", "neither"]
HATE_CLASS_ID = 0


def to_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df[["clean_text", "class_id"]].reset_index(drop=True))

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("class_id", "labels")
    keep = ["input_ids", "attention_mask", "labels"]
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def tune_hate_threshold(probs, y_true):
    # pick the P(hate) cut-off that maximises macro F1: above it we call the row
    # `hate`, otherwise fall back to the argmax over the remaining two classes.
    # tuned on validation only, never on test.
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.95, 0.01):
        y_pred = np.where(
            probs[:, HATE_CLASS_ID] >= threshold,
            HATE_CLASS_ID,
            np.argmax(probs[:, HATE_CLASS_ID + 1:], axis=1) + HATE_CLASS_ID + 1,
        )
        score = f1_score(y_true, y_pred, average="macro")
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_threshold, best_f1
