@echo off
setlocal enabledelayedexpansion
pushd %~dp0

set INPUT=data\samples\posts_sample.csv
set CONFIG=config\settings.mock.yaml
set OUTPUT=results\results_mock_posts_sample.csv

python -m src.run_batch --input %INPUT% --config %CONFIG% --output %OUTPUT%
if errorlevel 1 (
    echo Mock batch failed.
    popd
    exit /b 1
)

echo Starting Streamlit UI...
streamlit run app.py

popd
