import pandas as pd

from database_connection import PROJECT_ROOT
from preprocess import clean_text

PROC = PROJECT_ROOT / "data" / "processed"

SPLITS = ("train", "validation", "test")


def load_split(name):
    # the three splits are written by split_data.py as both .pkl and .csv;
    # the pickle keeps dtypes intact so we read that one.
    return pd.read_pickle(PROC / "{}.pkl".format(name)).reset_index(drop=True)


def load_train_augmented():
    # training set = the real train split plus the LLM-augmented hate/neither rows.
    # validation and test are never touched by augmentation, so they use load_split.
    train_df = load_split("train")

    aug_path = PROC / "augmented.csv"
    if not aug_path.exists():
        return train_df

    aug_df = pd.read_csv(aug_path)
    # the generated text is raw, so push it through the same cleaner as the splits
    # and drop anything that collapses to empty.
    aug_df["clean_text"] = aug_df["clean_text"].astype(str).map(clean_text)
    aug_df = aug_df[aug_df["clean_text"].str.len() > 0]
    aug_df = aug_df.reindex(columns=train_df.columns)

    merged = pd.concat([train_df, aug_df], ignore_index=True)
    print("merged {} augmented rows into the training set".format(len(aug_df)))
    return merged.reset_index(drop=True)
