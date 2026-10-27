.PHONY: setup install test physics benchmark lint format smoke check hooks train tune evaluate e1 fig-e1 docker-build docker-test

UV ?= uv
PYTHON ?= python3.11
UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
export UV_CACHE_DIR

setup:
	$(UV) sync --locked --group dev --python $(PYTHON)
install: setup
test:
	$(UV) run pytest
physics:
	$(UV) run pytest tests/test_point_mass.py tests/test_pipeline.py
benchmark:
	$(UV) run rocket-landing benchmark $(ARGS)
train:
	$(UV) run python scripts/train.py $(ARGS)
tune:
	$(UV) run python scripts/tune.py $(ARGS)
evaluate:
	$(UV) run python scripts/evaluate.py $(FIXTURE) $(ARGS)
e1:
	$(UV) run python scripts/run_experiment.py experiment=e1 $(ARGS)
fig-e1:
	$(UV) run python scripts/make_figures.py $(INPUTS) --output results/e1.png --kind learning
lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .
smoke:
	$(UV) run python scripts/smoke.py
check: lint test smoke
hooks:
	$(UV) run pre-commit install
docker-build:
	docker build -t rocket-landing-rl:local .
docker-test: docker-build
	docker run --rm rocket-landing-rl:local
