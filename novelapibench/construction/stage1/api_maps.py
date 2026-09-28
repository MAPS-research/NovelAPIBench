"""API maps: the public API surface of one library version (Appendix B.2, "Stage 1").

An introspection script runs inside the version's environment. It walks every importable
public submodule of the package (``pkgutil.walk_packages``; private, test, compat, setup and
version modules are skipped), and records each public function and class: fully qualified
name, signature and parameters, docstring, source file, and for classes their role
(exception, enum, protocol, typeddict, namedtuple, dataclass, plain_class). Submodules that
fail to import are retried after everything else has been imported (which resolves circular
imports), and those that still fail are recorded as ``walk_failures`` in the map's health
record: an API under such a subtree cannot be told apart from an API that is missing.

An object re-exported under several paths is kept once, at its best public path: a path
without private segments, closest public ancestor of the defining module, fewest segments.

Maps are cached per (package, version) under ``outputs/construction/api_maps/``, since every
boundary whose ``old_version`` lands on the same release shares it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

from omegaconf import DictConfig

from novelapibench.config import load_library_config
from novelapibench.construction.schemas import APIEntry, ParameterInfo
from novelapibench.log import logger
from novelapibench.paths import construction_dir

# Submodule-name fragments the walk never descends into.
WALK_SKIP_NAME_FRAGMENTS = (
    "._", ".tests", ".testing", ".test", ".compat", ".conftest", ".benchmarks",
    ".experimental._", ".distutils", ".setup", ".version",
)

_INTROSPECT_SCRIPT = textwrap.dedent(
    """
    import inspect
    import json
    import os
    import pkgutil
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    # Library imports may print to stdout; keep it clean for the JSON result.
    _real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    def _should_skip_submodule(name, skip_fragments):
        if name.startswith("_"):
            return True
        for frag in skip_fragments:
            if frag in name:
                return True
        return False

    def _safe_str(val):
        try:
            return str(val)
        except Exception:
            return None

    def _extract_params(sig):
        params = []
        for p in sig.parameters.values():
            params.append({
                "name": p.name,
                "annotation": _safe_str(p.annotation) if p.annotation is not inspect.Parameter.empty else None,
                "default": _safe_str(p.default) if p.default is not inspect.Parameter.empty else None,
                "kind": p.kind.name,
            })
        return params

    def _detect_class_role(obj):
        # Precedence: exception > enum > protocol > typeddict > namedtuple > dataclass > plain_class.
        try:
            if issubclass(obj, BaseException):
                return "exception"
        except TypeError:
            pass
        try:
            from enum import Enum
            if issubclass(obj, Enum):
                return "enum"
        except TypeError:
            pass
        if getattr(obj, "_is_protocol", False):
            return "protocol"
        if hasattr(obj, "__required_keys__") and hasattr(obj, "__total__"):
            return "typeddict"
        if hasattr(obj, "_fields") and isinstance(getattr(obj, "_field_defaults", None), dict):
            return "namedtuple"
        try:
            import dataclasses
            if dataclasses.is_dataclass(obj) and isinstance(obj, type):
                return "dataclass"
        except Exception:
            pass
        return "plain_class"

    def _introspect_obj(name, obj, module_name):
        entry = None
        try:
            owning = getattr(obj, "__module__", None)
        except Exception:
            owning = None
        try:
            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                try:
                    sig = inspect.signature(obj)
                    sig_str = str(sig)
                    params = _extract_params(sig)
                except (ValueError, TypeError):
                    sig_str = "()"
                    params = []
                entry = {
                    "api_name": f"{module_name}.{name}",
                    "kind": "function",
                    "module": module_name,
                    "owning_module": owning,
                    "signature": sig_str,
                    "parameters": params,
                    "docstring": inspect.getdoc(obj),
                    "source_file": inspect.getfile(obj) if hasattr(obj, "__code__") else None,
                }
            elif inspect.isclass(obj):
                try:
                    sig = inspect.signature(obj.__init__)
                except Exception:
                    sig = inspect.Signature()
                entry = {
                    "api_name": f"{module_name}.{name}",
                    "kind": "class",
                    "class_role": _detect_class_role(obj),
                    "module": module_name,
                    "owning_module": owning,
                    "signature": str(sig),
                    "parameters": _extract_params(sig),
                    "docstring": inspect.getdoc(obj),
                    "source_file": inspect.getfile(obj) if inspect.isclass(obj) else None,
                }
            elif callable(obj):
                # Other callables (ufuncs, functools.partial objects, ...).
                try:
                    sig = inspect.signature(obj)
                    sig_str = str(sig)
                    params = _extract_params(sig)
                except (ValueError, TypeError):
                    sig_str = "()"
                    params = []
                entry = {
                    "api_name": f"{module_name}.{name}",
                    "kind": "function",
                    "module": module_name,
                    "owning_module": owning,
                    "signature": sig_str,
                    "parameters": params,
                    "docstring": inspect.getdoc(obj),
                    "source_file": None,
                }
        except Exception as e:
            pass
        return entry

    def _walk_discover(roots, skip_fragments, failures=None, seen=None):
        # Expand package roots into their public submodules. Names pkgutil could not import
        # are appended to `failures` (walk_packages does not descend into them).
        discovered = []
        if seen is None:
            seen = set()
        for root_name in roots:
            try:
                root = __import__(root_name, fromlist=[""])
            except BaseException:
                if failures is not None:
                    failures.append(root_name)
                continue
            if root_name not in seen:
                discovered.append(root_name)
                seen.add(root_name)
            root_path = getattr(root, "__path__", None)
            if not root_path:
                continue
            try:
                walker = pkgutil.walk_packages(
                    root_path,
                    prefix=root_name + ".",
                    onerror=(failures.append if failures is not None
                             else (lambda _n: None)),
                )
            except Exception:
                continue
            while True:
                try:
                    info = next(walker)
                except StopIteration:
                    break
                except BaseException:
                    # A generator that raised cannot be resumed: the walk ends here.
                    if failures is not None:
                        failures.append(root_name + " <walk aborted>")
                    break
                name = info.name
                if _should_skip_submodule(name, skip_fragments):
                    continue
                if name in seen:
                    continue
                discovered.append(name)
                seen.add(name)
        return discovered

    def introspect_modules(module_names):
        results = {}
        for mod_name in module_names:
            try:
                mod = __import__(mod_name, fromlist=[""])
            except BaseException:
                continue
            entries = {}
            for name in dir(mod):
                if not name.startswith("_"):
                    try:
                        obj = getattr(mod, name)
                    except Exception:
                        continue
                    if obj is None:
                        continue
                    entry = _introspect_obj(name, obj, mod_name)
                    if entry:
                        entries[f"{mod_name}.{name}"] = entry
            results[mod_name] = entries
        return results

    payload = json.loads(sys.argv[1])
    modules = list(payload.get("modules", []))
    walk_from = list(payload.get("walk_from", []))
    skip_fragments = list(payload.get("skip_fragments", []))
    max_rounds = int(payload.get("walk_retry_rounds", 3))
    health = {"walk_failures": [], "recovered": [], "rounds": 0}
    if walk_from:
        _seen = set()
        failures = []
        discovered = _walk_discover(walk_from, skip_fragments, failures, _seen)
        # Retry rounds: importing everything that works first often resolves circular imports.
        for _round in range(max_rounds):
            if not failures:
                break
            health["rounds"] = _round + 1
            for _m in list(discovered):
                try:
                    __import__(_m, fromlist=[""])
                except BaseException:
                    pass
            retryable = [f for f in failures if not f.endswith(" <walk aborted>")]
            failures = []
            recovered_roots = []
            for _name in retryable:
                try:
                    __import__(_name, fromlist=[""])
                    recovered_roots.append(_name)
                except BaseException:
                    failures.append(_name)
            if not recovered_roots:
                break
            health["recovered"].extend(recovered_roots)
            discovered = discovered + _walk_discover(
                recovered_roots, skip_fragments, failures, _seen
            )
        health["walk_failures"] = sorted(set(failures))
        modules = modules + discovered
        _seen2 = set()
        _deduped = []
        for m in modules:
            if m not in _seen2:
                _deduped.append(m)
                _seen2.add(m)
        modules = _deduped
    output = introspect_modules(modules)
    health["n_modules_scanned"] = len(modules)
    health["n_modules_with_entries"] = sum(1 for v in output.values() if v)
    output["__health__"] = health

    sys.stdout = _real_stdout
    print(json.dumps(output))
    """
)


def introspect_modules_in_env(env_python: str, modules: list[str],
                              walk_from: list[str] | None = None, timeout: int = 300,
                              health: dict | None = None) -> dict[str, dict]:
    """Run the introspection script in ``env_python``: ``{module: {fq_name: entry}}``.

    ``health`` (if given) receives the walk record. stdout is parsed even on a non-zero exit
    code, since libraries may fail noisily on optional imports without spoiling the result.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_INTROSPECT_SCRIPT)
        script = f.name
    payload = {"modules": list(modules), "walk_from": list(walk_from or []),
               "skip_fragments": list(WALK_SKIP_NAME_FRAGMENTS)}
    try:
        r = subprocess.run([env_python, script, json.dumps(payload)], capture_output=True,
                           text=True, timeout=timeout)
        if r.stdout.strip():
            try:
                parsed = json.loads(r.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(f"introspection produced non-JSON stdout (rc={r.returncode}): "
                                   f"{r.stderr[:300]}")
            record = parsed.pop("__health__", None)
            if health is not None and isinstance(record, dict):
                health.update(record)
            return parsed
        raise RuntimeError(f"introspection produced no output (rc={r.returncode}): {r.stderr[:500]}")
    finally:
        os.unlink(script)


def _path_score(fq_name: str, scan_mod: str, owning_mod: str | None) -> tuple:
    """Lower is better: no private segment in the scan module; the scan module is an ancestor
    of the defining module, the closer the better; fewer segments; then lexicographic."""
    scan = scan_mod.split(".") if scan_mod else []
    owning = owning_mod.split(".") if owning_mod else []
    match = 0
    if owning:
        for a, b in zip(scan, owning):
            if a != b:
                break
            match += 1
        is_ancestor = match == len(scan)
    else:
        is_ancestor = True
    return (1 if any(s.startswith("_") for s in scan) else 0, 0 if is_ancestor else 1,
            -match if is_ancestor else 0, fq_name.count("."), fq_name)


def build_api_map(env_python: str, modules: list[str], library: str, version: str,
                  walk_from: list[str] | None = None, timeout: int = 300,
                  health: dict | None = None) -> dict[str, APIEntry]:
    """``{fq_name: APIEntry}`` of one environment, one entry per underlying object.

    Two scan locations are the same object when they agree on defining module, short name,
    signature and source file; the entry at the best-scoring path is kept.
    """
    raw = introspect_modules_in_env(env_python, modules, walk_from=walk_from, timeout=timeout,
                                    health=health)
    best: dict[tuple, tuple[tuple, str, str, dict]] = {}
    for mod_name, entries in raw.items():
        for fq_name, data in entries.items():
            key = (data.get("owning_module") or "", fq_name.rsplit(".", 1)[-1],
                   data.get("signature") or "", data.get("source_file") or "")
            score = _path_score(fq_name, mod_name, data.get("owning_module"))
            if key not in best or score < best[key][0]:
                best[key] = (score, fq_name, mod_name, data)
    api_map: dict[str, APIEntry] = {}
    for _score, fq_name, mod_name, data in best.values():
        extra = {"class_role": data["class_role"]} if "class_role" in data else {}
        api_map[fq_name] = APIEntry(
            api_name=fq_name, kind=data["kind"], module=mod_name,
            signature=data.get("signature", ""),
            parameters=[ParameterInfo(**p) for p in data.get("parameters", [])],
            docstring=data.get("docstring"), source_file=data.get("source_file"),
            library=library, old_version=version, new_version=version, extra=extra)
    return api_map


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def default_maps_dir() -> Path:
    return construction_dir() / "api_maps"


def api_map_path(maps_dir: Path, package: str, version: str) -> Path:
    return Path(maps_dir) / f"{package}__{version}.json"


def load_api_map(maps_dir: Path, package: str, version: str) -> dict[str, APIEntry] | None:
    p = api_map_path(maps_dir, package, version)
    if not p.exists():
        return None
    return {k: APIEntry(**v) for k, v in json.loads(p.read_text())["entries"].items()}


def load_map_health(maps_dir: Path, package: str, version: str) -> set[str]:
    """Subtrees the walk could not import (a truncated walk is not a lost subtree)."""
    p = api_map_path(maps_dir, package, version)
    if not p.exists():
        return set()
    health = json.loads(p.read_text()).get("health") or {}
    return {x for x in health.get("walk_failures", []) if not x.endswith(" <walk aborted>")}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_version_map(library: str, version: str, cfg: DictConfig, maps_dir: Path,
                      force: bool = False, repair: bool = True) -> Path:
    """Create the version's environment (repairing it if needed) and cache its API map."""
    from novelapibench.construction.envs import ensure_env, import_name, repair_env

    lib = load_library_config(library)
    out = api_map_path(maps_dir, str(lib.package), version)
    if out.exists() and not force:
        return out
    python = ensure_env(library, version)
    if repair:
        ok, note = repair_env(python, library, version)
        logger.info(f"{library}=={version}: environment {'healthy' if ok else 'NOT healthy'} ({note})")
    timeout = int(lib.get("construction", {}).get(
        "introspect_timeout_seconds", cfg.construction.stage1.introspect_timeout_seconds))
    t0 = time.monotonic()
    health: dict = {}
    api_map = build_api_map(python, [], library, version, walk_from=[import_name(library)],
                            timeout=timeout, health=health)
    payload = {"package": str(lib.package), "version": version, "library": library,
               "modules": [], "walk_from": [import_name(library)], "n_entries": len(api_map),
               "health": health, "entries": {k: v.model_dump() for k, v in api_map.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out)
    logger.info(f"{library}=={version}: {len(api_map)} entries, "
                f"{len(health.get('walk_failures', []))} walk failures ({time.monotonic() - t0:.0f}s)")
    return out
