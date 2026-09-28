"""Repository layout.

Everything the repository ships lives under ``configs/``, ``data/`` and ``envs/locks/``.
Everything the code produces goes under ``outputs/`` (override with ``NOVELAPIBENCH_OUTPUTS``);
the per-library execution environments go under ``envs/`` (override with ``NOVELAPIBENCH_ENVS``).
Scripts run with ``--debug`` write to ``outputs/debug/`` instead, so that quick runs on a few
tasks never mix with full runs.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"

# Released benchmark (read-only).
POOL_DIR = DATA_DIR / "pool"                  # frozen Stage-1 candidate pool
BENCHMARK_DIR = DATA_DIR / "benchmark"        # bundles, tasks, instances, splits, call records
PAPER_RESULTS_DIR = DATA_DIR / "paper_results"  # per-task verdicts behind the paper's figures

OUTPUTS_DIR = Path(os.environ.get("NOVELAPIBENCH_OUTPUTS", REPO_ROOT / "outputs"))
_outputs = OUTPUTS_DIR
ENVS_DIR = Path(os.environ.get("NOVELAPIBENCH_ENVS", REPO_ROOT / "envs"))
ENV_LOCKS_DIR = REPO_ROOT / "envs" / "locks"


def use_debug_outputs() -> None:
    """Send every output of this process to ``<outputs>/debug/``."""
    global _outputs
    _outputs = OUTPUTS_DIR / "debug"


def outputs_dir() -> Path:
    return _outputs


def runs_dir() -> Path:
    """Inference predictions and evaluation results: ``outputs/runs/<run>/<cell>/``."""
    return outputs_dir() / "runs"


def run_dir(run: str, cell: str | None = None) -> Path:
    d = runs_dir() / run
    return d / cell if cell else d


def instances_dir() -> Path:
    """Instances built for new backbones (``scripts/build_instance.py``): ``outputs/instances/<model>/``."""
    return outputs_dir() / "instances"


def adapters_dir() -> Path:
    """Trained adapters and edited checkpoints: ``outputs/adaptation/<method>/``."""
    return outputs_dir() / "adaptation"


def retrieval_index_dir(domain: str, condition: str) -> Path:
    return outputs_dir() / "retrieval_index" / domain / condition.replace("+", "_")


def construction_dir() -> Path:
    """Intermediate files of a benchmark (re)construction run."""
    return outputs_dir() / "construction"


def cache_dir() -> Path:
    return outputs_dir() / "cache"


def analysis_dir() -> Path:
    return outputs_dir() / "analysis"
