PY ?= ~/miniconda3/envs/personal/bin/python
PIP ?= ~/miniconda3/envs/personal/bin/pip
.PHONY: install data eval serve test
install:
	$(PIP) install -e ".[all]"
data:
	$(PY) -m recsys.data
eval:
	$(PY) -m recsys.pipeline
serve:
	$(PY) -m uvicorn api.main:app --reload --port 8000
test:
	$(PY) -m pytest -q
