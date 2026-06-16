PYTHON ?= python3
VENV ?= .venv

.PHONY: setup test demo compile

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements-test.txt

test:
	$(VENV)/bin/python -m pytest tests/unit -q
	npm run build

demo:
	$(VENV)/bin/python -m scripts.run_demo_pipeline

compile:
	$(VENV)/bin/python -m compileall -q .
