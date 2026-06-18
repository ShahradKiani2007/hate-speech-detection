import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

STEPS = [
    "import_to_db.py",
    "load_data.py",
    "preprocess.py",
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
