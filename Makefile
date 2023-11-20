SHELL:=/bin/zsh

.PHONY: lint
lint:
	poetry run flake8 **/*.py
	poetry run black --check --diff ./
	make mypy

.PHONY: mypy
mypy:
	poetry run mypy ./**/*.py

.PHONY: isort
isort:
	poetry run isort ./**/*.py

.PHONY: black
black:
	poetry run black ./

.PHONY: fix
fix:
	make isort black

.PHONY: install
install:
	poetry install