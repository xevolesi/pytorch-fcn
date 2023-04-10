SHELL := /bin/bash

.EXPORT_ALL_VARIABLES:
PYTHONPATH := ./
TEST_DIR := tests/
LINT_DIR := ./

lint:
	flake8 ${LINT_DIR}

# Call this to format your code.
format:
	isort .
	black .

verify_format:
	black --check .
	isort --check .

run_tests:
	pytest -svvv ${TEST_DIR}

reset_logs:
	rm -rf logs
	mkdir logs

# Call this before commit.
pre_push_test: verify_format lint run_tests