.PHONY: install test lint demo train run docker-up docker-down clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=roadlens --cov-report=term-missing

lint:
	flake8 roadlens/ server/ --count --select=E9,F63,F7,F82 --show-source --statistics

demo:
	python run_demo.py

# Train Chakranetra's own models (cost / degradation / repair failure) and
# regenerate the browser bundle. Trains on recorded repair costs when the
# database has enough of them, and on the labelled synthetic corpus when it
# does not -- see roadlens/ml/bootstrap.py.
train:
	python tools/train_models.py
	python tools/generate_ml_js.py

run:
	uvicorn server.app:app --reload

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ **/__pycache__ .pytest_cache coverage.xml htmlcov/
	rm -f roadlens.db
	rm -rf output/*
