# Iris MLOps (FastAPI + Docker)

## Quickstart (local)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py  # saves models/iris_clf.joblib and artifacts/metrics.json
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Browse: http://127.0.0.1:8000/docs

## Docker
docker build -t iris-api:latest .
docker run -p 8000:8000 iris-api:latest