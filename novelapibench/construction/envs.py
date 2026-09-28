"""Construction-time environments: one isolated environment per (package, version).

Stage 1 introspects every library version a release boundary needs (``new_version`` and one
``old_version`` per boundary); Stages 2-4 run generated code in the library's ``new_version``
environment. An environment is ``pip install <package>==<version>`` plus the package's optional
extras (except alternative backends, the catch-all ``all`` and dev/test/docs extras), in a
venv of the running interpreter when its Python version matches and in a conda environment
otherwise. Environments live under ``$NOVELAPIBENCH_ENVS/construction/<package>_<version>/``.

pip resolves dependencies to their newest releases, which can break an old library version
(it fails to import, or loses submodules during the Stage-1 walk). ``repair_env`` pins the
dependency the traceback blames to its last release on or before the library version's own
release date, one dependency at a time, and reverts a pin that shrinks the API map.

For Stages 2-4, ``stage_python`` prefers the locked execution environment
(``envs/<library>/``, the package set the released benchmark was built and evaluated in) and
falls back to a construction environment of ``new_version``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import venv
from functools import lru_cache
from pathlib import Path

from novelapibench.config import load_library_config
from novelapibench.log import logger
from novelapibench.paths import ENVS_DIR

# Extras never installed: alternative ML backends, dev/CI/docs tooling, catch-alls that pull
# in every backend, and deployment/distributed infrastructure.
SKIP_EXTRAS: frozenset[str] = frozenset({
    "tf", "tensorflow", "tensorflow-cpu", "tf-cpu", "tpu", "jax", "flax", "jax-cpu",
    "dev", "develop", "test", "tests", "testing", "docs", "doc", "quality", "lint",
    "benchmark", "benchmarks", "all", "full",
    "sagemaker", "deepspeed", "deepspeed-testing", "ray", "optuna", "codecarbon", "serving",
    "open-telemetry", "integrations",
})
_EXTRAS_MARKER = ".extras_installed"
_READY_MARKER = ".ready"


def envs_root() -> Path:
    return ENVS_DIR / "construction"


def env_name(package: str, version: str) -> str:
    return f"{package}_{version.replace('.', '_').replace('-', '_')}"


def env_python_of(env_dir: Path) -> str:
    for p in (env_dir / "bin" / "python", env_dir / "Scripts" / "python.exe"):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"no Python interpreter in {env_dir}")


def _installer_env() -> dict[str, str]:
    """Environment of pip/conda subprocesses: a private conda package cache (shared caches on
    network file systems have produced zero-byte bootstrap files), the stdlib distutils, and no
    user site-packages."""
    env = os.environ.copy()
    env.setdefault("CONDA_PKGS_DIRS", str(envs_root() / ".conda_pkgs"))
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    env["PYTHONNOUSERSITE"] = "1"
    Path(env["CONDA_PKGS_DIRS"]).mkdir(parents=True, exist_ok=True)
    return env


def _pip(python: str) -> list[str]:
    """pip of the running interpreter, installing into ``python``'s environment."""
    return [sys.executable, "-m", "pip", "--python", python]


def _check(python: str, code: str, what: str) -> None:
    r = subprocess.run([python, "-c", code], capture_output=True, text=True, env=_installer_env())
    if r.returncode != 0:
        raise RuntimeError(f"{what} failed in {python}: {(r.stderr or r.stdout)[:500]}")


def _verify(python: str, import_name: str | None = None) -> None:
    _check(python, "import importlib; importlib.import_module('pip._internal'); "
                   "importlib.import_module('packaging.version')", "bootstrap check")
    if import_name:
        _check(python, "import importlib.util, sys; "
                       f"sys.exit(0 if importlib.util.find_spec({import_name!r}) else 2)",
               f"import check of {import_name}")


def _conda_executable() -> str:
    for c in (os.environ.get("CONDA_EXE"), shutil.which("conda")):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("a different Python version is required but conda was not found")


def _create_base_env(env_dir: Path, python_version: str) -> None:
    want = ".".join(python_version.split(".")[:2])
    if want == f"{sys.version_info.major}.{sys.version_info.minor}":
        venv.EnvBuilder(with_pip=False, clear=True, symlinks=True).create(str(env_dir))
        return
    subprocess.run([_conda_executable(), "create", "--prefix", str(env_dir),
                    f"python={python_version}", "--yes", "--quiet"],
                   check=True, capture_output=True, env=_installer_env())


def _install_extras(env_dir: Path, package: str, version: str) -> None:
    """Install the package's optional extras one by one (a failing extra is skipped)."""
    if (env_dir / _EXTRAS_MARKER).exists():
        return
    python = env_python_of(env_dir)
    code = (f"import importlib.metadata, json; "
            f"print(json.dumps(importlib.metadata.metadata({package!r}).get_all('Provides-Extra') or []))")
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=15,
                           env=_installer_env())
        extras = json.loads(r.stdout.strip()) if r.returncode == 0 else []
    except Exception:  # noqa: BLE001
        extras = []
    for extra in (e for e in extras if e.lower() not in SKIP_EXTRAS):
        try:
            subprocess.run([*_pip(python), "install", "--quiet", f"{package}[{extra}]=={version}"],
                           check=True, capture_output=True, timeout=900, env=_installer_env())
            logger.info(f"installed extra [{extra}] for {package}=={version}")
        except (subprocess.SubprocessError, OSError):
            logger.debug(f"skipping extra [{extra}] for {package}=={version}")
    (env_dir / _EXTRAS_MARKER).touch()


def create_env_with_package(package: str, version: str, python_version: str = "3.11",
                            import_name: str | None = None,
                            extra_packages: list[str] | None = None,
                            root: Path | None = None) -> Path:
    """Create (or reuse) the environment ``<root>/<package>_<version>``.

    Creation runs under a file lock, and a ``.ready`` marker distinguishes a finished
    environment from one another process is still building or that a crash left half-built.
    """
    root = Path(root or envs_root())
    root.mkdir(parents=True, exist_ok=True)
    name = env_name(package, version)
    env_dir = root / name
    import_name = import_name or package
    with open(root / f".{name}.lock", "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if (env_dir / _READY_MARKER).exists():
                try:
                    _verify(env_python_of(env_dir), import_name)
                    _install_extras(env_dir, package, version)
                    return env_dir
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"{env_dir} failed validation ({exc}); rebuilding")
            if env_dir.exists():
                shutil.rmtree(env_dir, ignore_errors=True)
            logger.info(f"creating {env_dir} ({package}=={version}, Python {python_version})")
            _create_base_env(env_dir, python_version)
            python = env_python_of(env_dir)
            subprocess.run([*_pip(python), "install", "--quiet", "--upgrade", "pip", "setuptools",
                            "wheel", "packaging"], check=True, capture_output=True, env=_installer_env())
            _verify(python)
            subprocess.run([*_pip(python), "install", "--quiet", f"{package}=={version}",
                            *(extra_packages or [])], check=True, capture_output=True,
                           env=_installer_env())
            _verify(python, import_name)
            _install_extras(env_dir, package, version)
            (env_dir / _READY_MARKER).touch()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return env_dir


def import_name(library: str) -> str:
    lib = load_library_config(library)
    return str(lib.get("import_name") or lib.package)


def ensure_env(library: str, version: str) -> str:
    """Interpreter of the construction environment of ``library`` at ``version``."""
    lib = load_library_config(library)
    env_dir = create_env_with_package(str(lib.package), str(version),
                                      python_version=str(lib.get("python_version", "3.11")),
                                      import_name=import_name(library),
                                      extra_packages=list(lib.get("extra_packages") or []) or None)
    return env_python_of(env_dir)


_STAGE_PYTHON: dict[str, str] = {}
_STAGE_PYTHON_LOCK = threading.Lock()


def stage_python(library: str) -> str:
    """Interpreter that Stages 2-4 run a library's code in (its ``new_version``)."""
    from novelapibench.runtime.envs import EnvironmentNotReady, env_python
    with _STAGE_PYTHON_LOCK:
        if library not in _STAGE_PYTHON:
            try:
                _STAGE_PYTHON[library] = env_python(library)
            except (EnvironmentNotReady, FileNotFoundError) as exc:
                logger.info(f"{library}: locked execution environment unavailable ({exc}); "
                            "using a construction environment")
                _STAGE_PYTHON[library] = ensure_env(library, str(load_library_config(library).new_version))
        return _STAGE_PYTHON[library]


# ---------------------------------------------------------------------------
# Repair of environments broken by dependency drift
# ---------------------------------------------------------------------------

_SITE_RE = re.compile(r"site-packages/([A-Za-z0-9_.\-]+)/")


@lru_cache(maxsize=None)
def _pypi_releases(package: str) -> dict:
    req = urllib.request.Request(f"https://pypi.org/pypi/{package}/json",
                                 headers={"User-Agent": "novelapibench-construction/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["releases"]


def _is_prerelease(v: str) -> bool:
    low = v.lower()
    return any(t in low for t in ("a", "b", "rc", "dev")) and not low.replace(".", "").isdigit()


def _version_key(v: str) -> tuple:
    return tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in v.split("."))


def release_date(package: str, version: str) -> str | None:
    stamps = [f.get("upload_time_iso_8601") or f.get("upload_time")
              for f in _pypi_releases(package).get(version) or []]
    stamps = [s for s in stamps if s]
    return min(stamps)[:10] if stamps else None


def last_version_on_or_before(package: str, date: str) -> str | None:
    best = None
    for v, files in _pypi_releases(package).items():
        if not files or _is_prerelease(v):
            continue
        stamps = [f.get("upload_time_iso_8601") or f.get("upload_time") for f in files]
        stamps = [s for s in stamps if s]
        if stamps and min(stamps)[:10] <= date and (best is None or _version_key(v) > best[0]):
            best = (_version_key(v), v)
    return best[1] if best else None


def _run(python: str, args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([python, *args], capture_output=True, text=True, timeout=timeout)


def probe_import(python: str, module: str, timeout: int = 300) -> tuple[bool, str]:
    """``(ok, stderr)`` of importing ``module`` in the environment."""
    code = (f"import warnings, importlib\nwarnings.filterwarnings('ignore')\n"
            f"importlib.import_module({module!r})\n")
    try:
        r = _run(python, ["-c", code], timeout)
    except subprocess.TimeoutExpired:
        return False, "import probe timed out"
    return r.returncode == 0, r.stderr or ""


def probe_walk(python: str, library: str, root: str, timeout: int = 900) -> tuple[int, list[str]]:
    """``(n_entries, lost_subtrees)`` of a full Stage-1 walk from ``root``.

    A root import that succeeds proves little: a submodule can still fail to import (e.g. a
    removed NumPy function used by one subpackage), which silently drops its whole subtree.
    """
    from novelapibench.construction.stage1.api_maps import build_api_map
    health: dict = {}
    try:
        n = len(build_api_map(python, [], library, "probe", walk_from=[root], timeout=timeout,
                              health=health))
    except Exception as exc:  # noqa: BLE001
        return 0, [f"<introspection failed: {type(exc).__name__}>"]
    return n, [f for f in health.get("walk_failures", []) if not f.endswith(" <walk aborted>")]


def _blame(stderr: str, target: str, installed: set[str]) -> str | None:
    """The installed distribution most likely responsible for an import failure."""
    for pattern in (r"cannot import name '[^']+' from '([A-Za-z0-9_.]+)'",
                    r"No module named '([A-Za-z0-9_.]+)'"):
        m = re.search(pattern, stderr)
        if m and m.group(1).split(".")[0] != target:
            return m.group(1).split(".")[0]
    for name in reversed(_SITE_RE.findall(stderr)):   # deepest frame first
        top = name.split(".")[0]
        if top != target and top in installed:
            return top
    return None


def _installed(python: str) -> set[str]:
    out = set()
    for line in (_run(python, ["-m", "pip", "list", "--format=freeze"]).stdout or "").splitlines():
        if "==" in line:
            n = line.split("==")[0].strip()
            out |= {n, n.replace("-", "_"), n.replace("_", "-")}
    return out


def _distributions(python: str) -> dict[str, str]:
    """Top-level import name -> distribution name (e.g. ``mpl_toolkits`` -> ``matplotlib``)."""
    code = ("import json\nfrom importlib.metadata import packages_distributions\n"
            "print(json.dumps({k: v[0] for k, v in packages_distributions().items() if v}))\n")
    try:
        r = _run(python, ["-c", code])
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
    except Exception:  # noqa: BLE001
        return {}


def _installed_version(python: str, dist: str) -> str | None:
    for line in (_run(python, ["-m", "pip", "show", dist]).stdout or "").splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def repair_env(python: str, library: str, version: str, max_rounds: int = 5) -> tuple[bool, str]:
    """Pin blamed dependencies until the library imports and its walk loses no subtree.

    Only the dependency the traceback blames is pinned in each round (pinning every direct
    dependency drags unrelated packages to versions without wheels), and a pin that reduces the
    number of introspected APIs is reverted.
    """
    lib = load_library_config(library)
    package, module = str(lib.package), import_name(library)
    ok, stderr = probe_import(python, module)
    n, lost = probe_walk(python, library, module) if ok else (0, [])
    if ok and not lost:
        return True, f"clean ({n} entries)"
    rel = release_date(package, version)
    if not rel:
        return False, "release date unknown"
    installed, dists, notes = _installed(python), _distributions(python), []
    for _ in range(max_rounds):
        ok, stderr = probe_import(python, module)
        if ok:
            n, lost = probe_walk(python, library, module)
            if not lost:
                return True, "; ".join(notes + [f"clean ({n} entries)"])
            _, stderr = probe_import(python, lost[0])
            notes.append(f"walk lost {len(lost)} subtrees (e.g. {lost[0]})")
        dep = _blame(stderr, module, installed)
        if not dep:
            break
        dep = dists.get(dep, dep)
        try:
            pin = last_version_on_or_before(dep, rel)
        except Exception:  # noqa: BLE001  (not a PyPI distribution)
            break
        if not pin:
            break
        before, _ = probe_walk(python, library, module)
        prev = _installed_version(python, dep)
        r = _run(python, ["-m", "pip", "install", "--no-input", "-q", f"{dep}=={pin}"], timeout=3600)
        if r.returncode != 0:
            notes.append(f"pip install {dep}=={pin} failed")
            break
        after, _ = probe_walk(python, library, module)
        if after < before:
            _run(python, ["-m", "pip", "install", "--no-input", "-q", f"{dep}=={prev}" if prev else dep],
                 timeout=3600)
            notes.append(f"{dep}=={pin} reverted ({before} -> {after} entries)")
            break
        notes.append(f"{dep}=={pin} ({before} -> {after} entries)")
    ok, _ = probe_import(python, module)
    n, lost = probe_walk(python, library, module) if ok else (0, ["<import failed>"])
    return (ok and not lost), "; ".join(notes + [f"{n} entries, {len(lost)} lost subtrees"])
