import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

# prediction pipeline: make_predictions.py is self-contained — it loads the
# held-out tweets from the database, re-applies the saved preprocessing and
# feature transformers (no refit), loads the trained model, predicts, and
# writes the predictions back to the database.
STEPS = [
    "make_predictions.py",
]


def run(step):
    print("\n=== [predict] running {} ===".format(step))
    subprocess.run([sys.executable, str(SCRIPTS / step)], check=True, cwd=SCRIPTS)


def main():
    for step in STEPS:
        run(step)
    print("\nprediction pipeline finished — predictions written to database")


if __name__ == "__main__":
    main()
