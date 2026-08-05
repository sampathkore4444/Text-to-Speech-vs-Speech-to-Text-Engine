.PHONY: install install-engines dev lint lint-fix test run-api run-worker evaluate demo docker-up docker-down

install:
	python -m pip install -e ".[dev]"

install-engines:
	python -m pip install -e ".[dev,engines]"

lint:
	ruff check src tests scripts

lint-fix:
	ruff check --fix src tests scripts

test:
	pytest

run-api:
	uvicorn speechai.api.app:app --host 0.0.0.0 --port 8000 --reload

run-worker:
	python -m speechai.workers.batch_worker

evaluate:
	python -m speechai.cli.main evaluate data/manifest.jsonl

demo:
	python scripts/make_sample_audio.py && python -m speechai.cli.main transcribe data/samples/sample_01.wav

docker-up:
	docker compose up --build

docker-down:
	docker compose down
