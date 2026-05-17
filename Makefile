.PHONY: help install test lint format docs clean reproduce

help:
	@echo "PhysBench-1K Evaluation Framework"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  test         Run all tests"
	@echo "  lint         Run linters (black, flake8, mypy)"
	@echo "  format       Auto-format code with black"
	@echo "  docs         Build documentation"
	@echo "  reproduce    Reproduce all paper figures and tables"
	@echo "  clean        Clean generated files"
	@echo "  dataset      Validate dataset integrity"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=physbench --cov=prompts --cov=evaluation --cov=inference

lint:
	black --check physbench/ prompts/ evaluation/ inference/ tests/ scripts/
	flake8 physbench/ prompts/ evaluation/ inference/ tests/ --max-line-length=100
	mypy physbench/ prompts/ evaluation/ inference/ --ignore-missing-imports

format:
	black physbench/ prompts/ evaluation/ inference/ tests/ scripts/

docs:
	mkdocs serve

reproduce:
	python scripts/reproduce_all.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf build/ dist/ *.egg-info/

dataset:
	python -c "from physbench.validator import validate_example; from physbench import load_dataset; [validate_example(ex) for ex in load_dataset()[:10]]"
