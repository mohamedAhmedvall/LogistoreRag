.PHONY: help venv install qdrant-up qdrant-down qdrant-logs download-data ingest api ui test test-int eval lint format clean

# Détection automatique du venv local (.venv) — Windows et Unix
ifeq ($(OS),Windows_NT)
	VENV_BIN := .venv/Scripts
	VENV_PY := $(VENV_BIN)/python.exe
else
	VENV_BIN := .venv/bin
	VENV_PY := $(VENV_BIN)/python
endif

# Si le venv n'existe pas encore, fallback sur le python système (utile pour `make venv`).
PYTHON := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)
PIP := $(PYTHON) -m pip

help:
	@echo "Commandes disponibles :"
	@echo "  make venv           - Créer le venv local .venv/"
	@echo "  make install        - Installer les dépendances dans le venv"
	@echo "  make qdrant-up      - Démarrer Qdrant via Docker"
	@echo "  make qdrant-down    - Arrêter Qdrant"
	@echo "  make qdrant-logs    - Voir les logs Qdrant"
	@echo "  make download-data  - Télécharger le dataset Kaggle"
	@echo "  make ingest         - Lancer l'ingestion complète"
	@echo "  make api            - Démarrer l'API FastAPI (dev)"
	@echo "  make ui             - Démarrer le frontend Streamlit"
	@echo "  make test           - Lancer les tests unitaires"
	@echo "  make test-int       - Lancer les tests d'intégration (Qdrant requis)"
	@echo "  make eval           - Lancer la campagne d'évaluation"
	@echo "  make lint           - Vérifier le code (ruff + black)"
	@echo "  make format         - Formater le code"
	@echo "  make clean          - Nettoyer les artefacts"

venv:
	python -m venv .venv
	@echo "Venv créé. Activer avec: source .venv/Scripts/activate (Windows) ou source .venv/bin/activate (Unix)"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

qdrant-up:
	docker compose up -d qdrant
	@echo "Qdrant disponible sur http://localhost:6333 (dashboard: http://localhost:6333/dashboard)"

qdrant-down:
	docker compose down

qdrant-logs:
	docker compose logs -f qdrant

download-data:
	$(PYTHON) scripts/download_dataset.py

ingest:
	$(PYTHON) scripts/run_ingestion.py

api:
	$(PYTHON) -m uvicorn ragtime.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

test:
	$(PYTHON) -m pytest tests/unit -v --cov=src/ragtime --cov-report=term-missing

test-int:
	$(PYTHON) -m pytest tests/integration -v -m integration

eval:
	$(PYTHON) scripts/run_evaluation.py

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/ app/
	$(PYTHON) -m black --check src/ tests/ scripts/ app/

format:
	$(PYTHON) -m black src/ tests/ scripts/ app/
	$(PYTHON) -m ruff check --fix src/ tests/ scripts/ app/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
