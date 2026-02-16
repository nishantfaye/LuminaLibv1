.PHONY: up down build test lint format clean

up:
	docker-compose up --build

down:
	docker-compose down -v

build:
	docker-compose build

test:
	pytest tests/ -v

lint:
	mypy app/
	flake8 app/

format:
	black app/ tests/
	isort app/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache
