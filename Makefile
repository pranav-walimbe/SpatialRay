.DEFAULT_GOAL := fix

.PHONY: sync fix lint test ci perf-preprocess

sync:
	uv sync --group dev

fix:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest || [ $$? -eq 5 ]

ci: lint test

perf-preprocess:
	uv run python -m perf.preprocess
