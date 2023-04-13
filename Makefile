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
	rm -rf wandb

export_to_onnx:
	python export_to_onnx.py \
		--config config.yml \
		--torch_weights logs/2023-04-12\ 08\:04\:42.694413/weights/fcn_8_iou_0.6438923478126526.pt \
		--onnx_path ./fcn.onnx \
		--image_size 500,500 \
		--do_check_on_validation_set

# Call this before commit.
pre_push_test: verify_format lint run_tests