import json
import os
import subprocess


def test_training_produces_model_and_metrics():
    # Run training script
    subprocess.check_call(["python", "src/train.py"])  # tiny + deterministic

    assert os.path.exists("models/iris_clf.joblib")
    assert os.path.exists("artifacts/metrics.json")

    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)

    # sanity threshold for iris
    assert metrics["accuracy"] >= 0.9