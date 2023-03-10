SHELL := /bin/bash

.EXPORT_ALL_VARIABLES:
PYTHONPATH := ./
TEST_DIR := tests/
LINT_DIR := ./

lint:
	flake8 --test-func-name-validator-regex="test_.*" ${LINT_DIR}

# Call this to format your code.
format:
	isort .
	black .

verify_format:
	black --check .
	isort --check .

run_tests:
	pytest -svvv ${TEST_DIR}

# Call this before commit.
pre_push_test: verify_format lint run_tests