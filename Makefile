.PHONY: help install dev up down logs migrate lint format test check clean

# Default target
help:
	@echo "AI Shorts Engine - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install production dependencies"
	@echo "  dev         Install development dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  up          Start all services with Docker Compose"
	@echo "  down        Stop all services"
	@echo "  logs        Tail logs from all services"
	@echo "  build       Rebuild Docker images"
	@echo ""
	@echo "Database:"
	@echo "  migrate     Run database migrations"
	@echo "  migration   Create a new migration (usage: make migration name=add_users)"
	@echo ""
	@echo "Quality:"
	@echo "  lint        Run linters (ruff, mypy)"
	@echo "  format      Format code (black, isort)"
	@echo "  test        Run test suite"
	@echo "  check       Run all checks (format, lint, test)"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Remove build artifacts and caches"
	@echo "  shell       Open a shell in the API container"
	@echo "  smoke       Run smoke test"

# Setup
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install || true

# Docker
up:
	docker compose up -d
	@echo "Services started. API available at http://localhost:8000"

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build --no-cache

# Database
migrate:
	docker compose exec api alembic upgrade head

migrate-local:
	alembic upgrade head

migration:
	@if [ -z "$(name)" ]; then \
		echo "Usage: make migration name=migration_name"; \
		exit 1; \
	fi
	docker compose exec api alembic revision --autogenerate -m "$(name)"

migration-local:
	@if [ -z "$(name)" ]; then \
		echo "Usage: make migration-local name=migration_name"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(name)"

# Quality
lint:
	ruff check src tests
	mypy src

format:
	black src tests
	isort src tests
	ruff check --fix src tests

test:
	pytest tests/ -v --cov=shorts_engine --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short

check: format lint test
	@echo "All checks passed!"

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/

shell:
	docker compose exec api /bin/bash

smoke:
	curl -X POST http://localhost:8000/api/v1/jobs/smoke

smoke-cli:
	shorts-engine smoke
