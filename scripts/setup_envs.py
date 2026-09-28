"""Create the per-library execution environments from envs/locks/.

Generated code for a task runs in an environment with the library's new version. Each
environment is a Python 3.11 venv under envs/<library>/ (set NOVELAPIBENCH_ENVS to put them
elsewhere) with the exact package set pinned in envs/locks/<library>.txt. Four libraries (pandas,
scipy, rdkit, httpx) run in the main environment and need nothing here.

Examples:
    python scripts/setup_envs.py                            # all libraries (~40 GB of packages)
    python scripts/setup_envs.py --libraries flask pydantic numpy
    python scripts/setup_envs.py --check                    # report which environments are ready
    python scripts/setup_envs.py --for-instance qwen2.5-coder-7b --domains swe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelapibench.benchmark import load_instance  # noqa: E402
from novelapibench.config import list_libraries  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.runtime.envs import EnvironmentNotReady, create_env, env_python  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libraries", nargs="+", default=None)
    ap.add_argument("--for-instance", default=None, help="only the libraries of this backbone's instance")
    ap.add_argument("--domains", nargs="+", default=None, help="with --for-instance: restrict to domains")
    ap.add_argument("--check", action="store_true", help="only report status")
    ap.add_argument("--force", action="store_true", help="recreate existing environments")
    args = ap.parse_args()
    setup_logging()

    libs = args.libraries or list_libraries()
    if args.for_instance:
        wanted = {t.library for t in load_instance(args.for_instance, args.domains)}
        libs = [lib for lib in libs if lib in wanted]
    missing = []
    for lib in libs:
        if not args.check:
            try:
                create_env(lib, force=args.force)
            except Exception as e:  # noqa: BLE001 — report and continue with the next library
                logger.error(f"{lib}: {e}")
        try:
            logger.info(f"{lib:14s} ready  ({env_python(lib)})")
        except EnvironmentNotReady as e:
            missing.append(lib)
            logger.warning(f"{lib:14s} not ready: {e}")
    if missing:
        raise SystemExit(f"not ready: {' '.join(missing)}")


if __name__ == "__main__":
    main()
