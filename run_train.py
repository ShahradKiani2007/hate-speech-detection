import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

# training pipeline: data prep (phase 2) -> split -> model training
STEPS = [
    "import_to_db.py",
    "load_data.py",
    "preprocess.py",
    "split_data.py",
    "feature_engineering.py",
    "train_model.py",
]


def run(step):
    print("\n=== [train] running {} ===".format(step))
    subprocess.run([sys.executable, str(SCRIPTS / step)], check=True, cwd=SCRIPTS)


def main():
    for step in STEPS:
        run(step)
    print("\ntraining pipeline finished — model saved to models/")


if __name__ == "__main__":
    main()
