import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from database_connection import PROJECT_ROOT

SEED = 42
PROC = PROJECT_ROOT / "data" / "processed"


def main():
    df = pd.read_pickle(PROC / "clean.pkl")

    # 70 / 15 / 15 train / val / test. keeping this two-step stratified split (and
    # seed) identical to the notebook is what lets us reuse the pre-generated
    # augmented.csv without leaking augmented hate rows into val / test.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["class_id"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["class_id"], random_state=SEED
    )

    for name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        part = part.reset_index(drop=True)
        part.to_pickle(PROC / "{}.pkl".format(name))
        part.to_csv(PROC / "{}.csv".format(name), index=False)
        counts = np.bincount(part["class_id"].to_numpy(), minlength=3).tolist()
        print("  {:11s} {:5d} rows  class counts (hate/off/neither): {}".format(
            name, len(part), counts))


if __name__ == "__main__":
    main()
