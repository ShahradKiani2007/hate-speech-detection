import subprocess
import sys
from pathlib import Path

from prefect import flow, task

SCRIPTS = Path(__file__).resolve().parent / "scripts"


@task(retries=1)
def run_step(step):
    subprocess.run([sys.executable, str(SCRIPTS / step)], check=True, cwd=SCRIPTS)
    return step


@flow(name="train")
def train_flow():
    a = run_step("import_to_db.py")
    b = run_step("load_data.py", wait_for=[a])
    c = run_step("preprocess.py", wait_for=[b])
    d = run_step("split_data.py", wait_for=[c])
    e = run_step("make_augmentations.py", wait_for=[d])
    f = run_step("feature_engineering.py", wait_for=[e])
    run_step("train_model.py", wait_for=[f])


@flow(name="predict")
def predict_flow():
    run_step("make_predictions.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["train", "predict"], nargs="?", default="train")
    args = parser.parse_args()
    (train_flow if args.which == "train" else predict_flow)()
