import os
from datetime import datetime

import torch
from clearml import Task

from source.utils.general import read_config, seed_everything
from source.utils.training import train

# Set benchmark to True and deterministic to False
# if you want to speed up training with less level of reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Speed up GEMM if GPU allowed to use TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    config = read_config("config.yml")
    if config.training.use_clearml:
        task = Task.init(project_name="FCN")
        task.connect(config)
        cm_logger = task.get_logger()
    else:
        cm_logger = None

    # Create folders for current run.
    os.makedirs(config.logs.log_dir, exist_ok=True)
    current_run_log_path = os.path.join(config.logs.log_dir, f"{datetime.now()}")
    os.makedirs(current_run_log_path, exist_ok=True)
    os.makedirs(
        os.path.join(current_run_log_path, config.logs.weights_folder), exist_ok=True
    )
    os.makedirs(
        os.path.join(current_run_log_path, config.logs.fixed_batch_predictions),
        exist_ok=True,
    )

    seed_everything(config)
    train(config, current_run_log_path, cm_logger)


if __name__ == "__main__":
    main()
