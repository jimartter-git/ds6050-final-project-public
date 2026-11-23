import json
import os
import random
import subprocess
import numpy as np
import torch


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


class RunLogger:
    def __init__(self, directory):
        self.directory = directory

    def log(self, values):
        with (self.directory / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(values, allow_nan=False) + '\n')
        print(json.dumps(values, allow_nan=False))
