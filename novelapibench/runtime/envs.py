"""Per-library execution environments.

Generated code for a task runs in an environment that has the library's new version
(``new_version`` in ``configs/libraries/<library>.yaml``). The exact package sets used for the
paper are pinned in ``envs/locks/<library>.txt`` (``pip freeze`` of the environment that ran the
evaluation). A lockfile whose first line is ``# main-environment`` means the library ran in the
main project environment (whose ``requirements.txt`` pins the same versions).

``scripts/setup_envs.py`` creates the environments under ``envs/<library>/``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from novelapibench.config import load_library_config
from novelapibench.log import logger
from novelapibench.paths import ENV_LOCKS_DIR, ENVS_DIR

MAIN_ENV_MARKER = "# main-environment"


class EnvironmentNotReady(RuntimeError):
    pass


def lockfile(library: str) -> Path:
    return ENV_LOCKS_DIR / f"{library}.txt"


def uses_main_env(library: str) -> bool:
    path = lockfile(library)
    return path.exists() and path.read_text().startswith(MAIN_ENV_MARKER)


def env_dir(library: str) -> Path:
    return ENVS_DIR / library


def _installed_version(python: str, package: str) -> str | None:
    code = f"import importlib.metadata as m; print(m.version({package!r}))"
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


@lru_cache(maxsize=None)
def env_python(library: str) -> str:
    """Interpreter of the library's execution environment (checked once per process)."""
    lib = load_library_config(library)
    if uses_main_env(library):
        python = sys.executable
    else:
        python = str(env_dir(library) / "bin" / "python")
        if not Path(python).exists():
            raise EnvironmentNotReady(
                f"no execution environment for {library}; run "
                f"`python scripts/setup_envs.py --libraries {library}`")
    have = _installed_version(python, lib.package)
    if have != str(lib.new_version):
        raise EnvironmentNotReady(
            f"{python} has {lib.package}=={have}, expected {lib.new_version} "
            f"(see envs/locks/{library}.txt)")
    return python


def create_env(library: str, force: bool = False) -> Path:
    """Create ``envs/<library>/`` from its lockfile (a venv of the running Python 3.11)."""
    lib = load_library_config(library)
    if uses_main_env(library):
        logger.info(f"{library}: runs in the main environment, nothing to create")
        return Path(sys.executable).parent.parent
    want = str(lib.get("python_version", "3.11"))
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    if want != have:
        raise RuntimeError(f"{library} needs Python {want}; run setup_envs.py with that interpreter")
    target = env_dir(library)
    python = target / "bin" / "python"
    if python.exists() and not force:
        env_python.cache_clear()
        try:
            env_python(library)
            logger.info(f"{library}: {target} exists")
            return target
        except EnvironmentNotReady:
            logger.info(f"{library}: {target} is incomplete; recreating it")
    lock = lockfile(library)
    if not lock.exists():
        raise FileNotFoundError(lock)
    logger.info(f"{library}: creating {target}")
    subprocess.run([sys.executable, "-m", "venv", "--clear", str(target)], check=True)
    pip = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    subprocess.run(pip + ["--upgrade", "pip"], check=True)
    # --no-deps: the lockfile is a complete, exact package set.
    env = dict(os.environ, PIP_NO_CACHE_DIR=os.environ.get("PIP_NO_CACHE_DIR", "0"))
    subprocess.run(pip + ["--no-deps", "-r", str(lock)], check=True, env=env)
    env_python.cache_clear()
    env_python(library)  # verify
    return target
