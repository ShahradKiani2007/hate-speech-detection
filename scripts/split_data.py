import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from database_connection import PROJECT_ROOT

SEED = 42
PROC = PROJECT_ROOT / "data" / "processed"

# 70 / 15 / 15 train / val / test
TEST_FRAC = 0.15
VAL_FRAC = 0.15


def main():
    df = pd.read_pickle(PROC / "clean.pkl")

    y = df["class_id"].to_numpy()
    tweet_ids = df["tweet_id"].to_numpy()

    idx = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        idx, test_size=TEST_FRAC, random_state=SEED, stratify=y
    )
    # val fraction is relative to the whole set, so rescale for the second split
    val_ratio = VAL_FRAC / (1 - TEST_FRAC)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=SEED, stratify=y[train_val_idx]
    )

    np.savez(
        PROC / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        tweet_ids=tweet_ids,
    )

    print(
        "split rows -> train {}, val {}, test {}".format(
            len(train_idx), len(val_idx), len(test_idx)
        )
    )
    for name, part in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = np.bincount(y[part], minlength=3)
        print("  {:5s} class counts (hate/off/neither): {}".format(name, counts.tolist()))


if __name__ == "__main__":
    main()
