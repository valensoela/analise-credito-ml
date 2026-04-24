.PHONY: setup train train-xgb api streamlit mlflow docker-up docker-down test lint clean

# SETUP
setup:
	uv sync

# TREINAMENTO DO MODELO
train:
	uv run python src/train.py --model all

train-xgb:
	uv run python src/train.py --model xgboost

# MLFLOW, API E STREAMLIT
mlflow:
	uv run mlflow ui --host 127.0.0.1 --port 5000

api:
	uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

streamlit:
	uv run streamlit run app/streamlit_app.py

# DOCKER
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# TESTES
test:
	uv run pytest tests/ \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html \
        --html=reports/test_report.html \
        --self-contained-html \
        -v
		
lint:
	uv run ruff check src/ api/ app/ tests/

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml
	rm -rf mlruns/
