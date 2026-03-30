.PHONY: install test generate run-cc run-pmcc clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

generate:
	python -m src.data.sample_data --ticker AAPL --start 2023-01-01 --end 2024-01-01

run-cc:
	python -m src.main --config configs/covered_call_default.yaml

run-pmcc:
	python -m src.main --config configs/pmcc_default.yaml

clean:
	rm -rf data/processed/*.parquet output*.html __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
