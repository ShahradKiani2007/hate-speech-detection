import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

# data-prep only (no model training). feature_engineering now reads the train/val
# splits, so split_data + make_augmentations run first (the latter reuses the
# committed augmented.csv offline).
STEPS = [
    "import_to_db.py",
    "load_data.py",
    "preprocess.py",
    "split_data.py",
    "make_augmentations.py",
    "feature_engineering.py",
]


def run(step):
    print(f"\n=== running {step} ===")
    subprocess.run([sys.executable, str(SCRIPTS / step)], check=True, cwd=SCRIPTS)


def main():
    for step in STEPS:
        run(step)
    print("\npipeline finished")


if __name__ == "__main__":
    main()
