PYTHON ?= python
STREAMLIT ?= streamlit
INPUT_DATA := data/samples/posts_sample.csv
MOCK_CONFIG := config/settings.mock.yaml
MOCK_OUTPUT := results/results_mock_posts_sample.csv

.PHONY: setup test mock ui clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

mock:
	$(PYTHON) -m src.run_batch --input $(INPUT_DATA) --config $(MOCK_CONFIG) --output $(MOCK_OUTPUT)

ui:
	$(STREAMLIT) run app.py

clean:
	$(PYTHON) -c "import shutil, pathlib; base = pathlib.Path('.');\
for path in list(base.rglob('__pycache__')): \
    shutil.rmtree(path, ignore_errors=True); \
results = base / 'results'; \
results.mkdir(parents=True, exist_ok=True); \
for pattern in ('results_mock_*.csv', 'report_results_mock_*.md'): \
    for f in results.glob(pattern): \
        f.unlink(missing_ok=True)"
