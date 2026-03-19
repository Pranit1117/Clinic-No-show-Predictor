.PHONY: help setup install lint test run-api run-app mlflow-ui clean

help:
	@echo "═══════════════════════════════════════════════════"
	@echo "  Clinic No-Show Predictor — Command Reference"
	@echo "═══════════════════════════════════════════════════"
	@echo "  make setup        Create venv & install deps"
	@echo "  make install      Install deps into active venv"
	@echo "  make lint         Run ruff linter"
	@echo "  make test         Run pytest with coverage"
	@echo "  make ingest       Download & validate raw data"
	@echo "  make features     Build feature set"
	@echo "  make train        Train all models + log to MLflow"
	@echo "  make run-api      Start FastAPI server (port 8000)"
	@echo "  make run-app      Start Streamlit dashboard"
	@echo "  make mlflow-ui    Open MLflow experiment tracker"
	@echo "  make clean        Remove caches and artifacts"
	@echo "═══════════════════════════════════════════════════"

setup:
	python -m venv .venv
	@echo "Run: source .venv/bin/activate"
	@echo "Then: make install"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check src/ tests/ prefect_flows/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

ingest:
	python -m src.data.ingest

features:
	python -m src.features.build_features

train:
	python -m src.models.train

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-app:
	streamlit run deployment/streamlit/app.py

mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
