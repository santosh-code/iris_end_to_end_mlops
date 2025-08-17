from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json


def main(random_state: int = 42, test_size: float = 0.2, model_path: str = "models/iris_clf.joblib"):
    # Load Iris
    X, y = load_iris(return_X_y=True, as_frame=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Save model & metrics
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=2)

    print(f"saved model to {model_path} with acc={acc:.4f}")


if __name__ == "__main__":
    main()