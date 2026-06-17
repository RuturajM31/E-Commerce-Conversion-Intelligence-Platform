# Makefile for E-Commerce Conversion Intelligence Platform

.PHONY: install test syntax app baseline thresholds model-selection final-champion forecasting anomaly mvd all-pipelines

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

syntax:
	python3 -m compileall app src tests

test:
	pytest -q

app:
	python3 -m streamlit run app/Executive_Overview.py

baseline:
	python3 -m src.models.train_baseline_model

thresholds:
	python3 -m src.models.analyze_thresholds

model-selection:
	python3 -m src.models.run_model_selection

final-champion:
	python3 -m src.models.finalize_true_champion

forecasting:
	python3 -m src.forecasting.build_business_forecasts

anomaly:
	python3 -m src.anomaly.build_anomaly_signals

mvd:
	python3 -m src.models.build_mvd_method_coverage

all-pipelines:
	python3 -m src.models.run_model_selection
	python3 -m src.models.finalize_true_champion
	python3 -m src.forecasting.build_business_forecasts
	python3 -m src.anomaly.build_anomaly_signals
	python3 -m src.models.build_mvd_method_coverage
