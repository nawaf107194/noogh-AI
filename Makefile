.PHONY: install run test clean init-db

VENV_BIN = venv/bin
PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/pip
UVICORN = $(VENV_BIN)/uvicorn
PYTEST = $(VENV_BIN)/pytest

install:
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTEST)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

init-db:
	$(PYTHON) -m src.core.init_db

verify:
	$(PYTHON) verify_app.py
	$(PYTHON) tests/test_di_services.py
