from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/iris_clf.joblib")
app = FastAPI(title="Iris Classifier API")

class IrisIn(BaseModel):
    sepal_length: float = Field(..., description="sepal length (cm)")
    sepal_width: float = Field(..., description="sepal width (cm)")
    petal_length: float = Field(..., description="petal length (cm)")
    petal_width: float = Field(..., description="petal width (cm)")

class PredictionOut(BaseModel):
    label: str
    label_id: int
    probabilities: dict

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    global target_names
    target_names = ["setosa", "versicolor", "virginica"]

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(inp: IrisIn):
    X = np.array([[
        inp.sepal_length,
        inp.sepal_width,
        inp.petal_length,
        inp.petal_width,
    ]])
    probs = model.predict_proba(X)[0]
    label_id = int(np.argmax(probs))
    return {
        "label": target_names[label_id],
        "label_id": label_id,
        "probabilities": {target_names[i]: float(probs[i]) for i in range(len(probs))}
    }