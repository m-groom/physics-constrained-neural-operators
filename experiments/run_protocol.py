"""Run protocol: seeding, per-run training logs and run manifests.

Shared by the training and evaluation scripts so that a run is reproducible from
its seed and traceable from the artefacts it leaves behind.
"""

import csv
import json
import math
import os
import random
import subprocess
from datetime import UTC, datetime

import numpy as np
import torch


def seed_everything(seed):
    """Seed Python, NumPy and torch (CPU and CUDA) from one integer.

    Also asks cuDNN for deterministic algorithms, which costs a little
    throughput on convolutional models and nothing on the spectral ones.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _seed_worker(worker_id):
    """Seed a DataLoader worker from the seed torch handed it."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_kwargs(seed):
    """DataLoader keyword arguments that make shuffling and workers reproducible."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return {"generator": generator, "worker_init_fn": _seed_worker}


def start_training_log(path, columns, resume=False):
    """Start the per-run training log, truncating it unless this run is a resume.

    A fresh run starts a new file with a header row; ``resume=True`` keeps an
    existing file and appends to it. Appending across independent runs is what
    spliced two different configurations into one log in the March-April runs.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if resume and os.path.exists(path):
        return
    with open(path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()


def best_logged_eval_loss(path, column="eval_l2"):
    """Smallest finite value of ``column`` in an existing training log.

    Returns ``math.inf`` when the log is missing or holds no finite entry, so a
    resumed run keeps the best validation loss of the run it continues.
    """
    if not os.path.exists(path):
        return math.inf
    best = math.inf
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                best = min(best, value)
    return best


def git_revision(repo_dir):
    """Current git commit of ``repo_dir``, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_run_manifest(save_dir, config, seed, checkpoint_paths, extra=None):
    """Write ``run_manifest.json`` beside the checkpoints of a run.

    Records the resolved configuration, the seed, the git commit, the start
    time, the SLURM job id and the checkpoint paths, so a result on disk can be
    traced back to the code and configuration that produced it.
    """
    os.makedirs(save_dir, exist_ok=True)
    manifest = {
        "config": config,
        "seed": int(seed),
        "git_commit": git_revision(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "start_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint_paths": list(checkpoint_paths),
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(save_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return path


def check_run_directory(paths, overwrite=False, resume=False):
    """Refuse to start a fresh run that would overwrite existing artefacts.

    ``paths`` are the files this run will write (checkpoint, best checkpoint,
    training log). A resume expects them; a fresh run must be given
    ``overwrite=True`` before it may replace them.
    """
    if overwrite or resume:
        return
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "this run would overwrite "
            + ", ".join(sorted(existing))
            + "; pass --overwrite to replace them or --resume_training to continue the run"
        )
