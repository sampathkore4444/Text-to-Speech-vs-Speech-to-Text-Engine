.PHONY: install install-engines dev lint lint-fix test run-api run-worker evaluate demo verify-model docker-up docker-down

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

# --- Model promotion gate (CI-agnostic) ------------------------------------
# Verify an int8 CTranslate2 export against the fp32 baseline before pointing
# stt.model_path at it: quantization gap, absolute WER/RTF bars, and the
# batch-job + WebSocket serving-path spot checks. The defaults below are
# *stricter* than the script's dev defaults (gap 0.05 / WER 0.10) - the
# promotion bar. Override any variable on the command line or via the env.
#   make verify-model CT2_DIR=data/models/finetuned-v2/ct2 MAX_WER_GAP=0.01
# GitHub equivalent: .github/workflows/model-promotion.yml
CT2_DIR ?= data/models/finetuned/ct2
EVAL_MANIFEST ?= data/eval_manifest.jsonl
FP32_REPORT ?= data/models/finetuned/report_finetuned.json
MODEL_LANGUAGE ?= en
MAX_WER_GAP ?= 0.02
MAX_WER_ABS ?= 0.08
MAX_RTF ?= 0.50
VERIFY_REPORT ?= data/eval/verify_ct2-promotion.json

verify-model:
	python scripts/verify_ct2_model.py \
		--ct2-dir "$(CT2_DIR)" \
		--manifest "$(EVAL_MANIFEST)" \
		--fp32-report "$(FP32_REPORT)" \
		--language "$(MODEL_LANGUAGE)" \
		--max-wer-gap "$(MAX_WER_GAP)" \
		--max-wer-abs "$(MAX_WER_ABS)" \
		--max-rtf "$(MAX_RTF)" \
		--report "$(VERIFY_REPORT)" \
		--no-mlflow

docker-up:
	docker compose up --build

docker-down:
	docker compose down
