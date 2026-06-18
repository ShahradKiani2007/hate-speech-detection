import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from load_data import load_data  # noqa: E402
from preprocess import clean_text  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
STATS = Path(__file__).resolve().parent / "eda_stats.txt"

LABELS = {0: "hate", 1: "offensive", 2: "neither"}
PALETTE = {"hate": "#c0392b", "offensive": "#e67e22", "neither": "#27ae60"}
STOP = set("the a an and or to of in is it for on with you that this be are i my we so".split())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df["label"] = df["class_id"].map(LABELS)
    df["clean"] = df["text"].astype(str).apply(clean_text)
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["clean"].str.split().str.len()
    sns.set_theme(style="whitegrid")

    out = []

    def log(s):
        print(s)
        out.append(s)

    log(f"rows: {len(df)}")
    log("class distribution:")
    log(df["label"].value_counts().to_string())
    log("class share (%):")
    log((df["label"].value_counts(normalize=True) * 100).round(2).to_string())

    # 1. class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["hate", "offensive", "neither"]
    sns.countplot(data=df, x="label", hue="label", order=order, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("")
    save(fig, "01_class_distribution.png")

    # 2. tweet length by class
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="label", y="word_count", hue="label", order=order, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Word count by class (cleaned text)")
    ax.set_ylim(0, 35)
    save(fig, "02_wordcount_by_class.png")
    log("\nword_count by class (mean):")
    log(df.groupby("label")["word_count"].mean().round(2).to_string())

    # 3. char count distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["char_count"], bins=40, color="#2c3e50", ax=ax)
    ax.set_title("Raw tweet length (characters)")
    save(fig, "03_charcount_hist.png")
    log("\nchar_count describe:")
    log(df["char_count"].describe().round(2).to_string())

    # 4. annotator agreement
    df["max_vote"] = df[["hate_votes", "offensive_votes", "neither_votes"]].max(axis=1)
    df["agreement"] = df["max_vote"] / df["annotator_count"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="agreement", hue="label", multiple="stack",
                 palette=PALETTE, bins=20, ax=ax)
    ax.set_title("Annotator agreement (top votes / total)")
    save(fig, "04_annotator_agreement.png")
    log("\nmean annotator agreement by class:")
    log(df.groupby("label")["agreement"].mean().round(3).to_string())
    log(f"\nfully unanimous tweets: {(df['agreement'] == 1).mean() * 100:.1f}%")

    # 5. top tokens per class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, lab in zip(axes, order):
        toks = Counter()
        for t in df.loc[df["label"] == lab, "clean"]:
            toks.update(w for w in t.split() if w not in STOP and len(w) > 2)
        common = toks.most_common(12)
        words = [w for w, _ in common][::-1]
        counts = [c for _, c in common][::-1]
        ax.barh(words, counts, color=PALETTE[lab])
        ax.set_title(f"Top words: {lab}")
    save(fig, "05_top_words_by_class.png")

    # 6. structural feature correlation
    feat = pd.DataFrame({
        "char_count": df["char_count"],
        "word_count": df["word_count"],
        "exclaim": df["text"].str.count("!"),
        "question": df["text"].str.count(r"\?"),
        "mention": df["text"].str.count("@"),
        "hashtag": df["text"].str.count("#"),
        "uppercase": df["text"].str.count(r"[A-Z]"),
    })
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(feat.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Structural feature correlation")
    save(fig, "06_feature_correlation.png")

    STATS.write_text("\n".join(out))
    print(f"\nfigures saved to {FIG}, stats -> {STATS.name}")


if __name__ == "__main__":
    main()
