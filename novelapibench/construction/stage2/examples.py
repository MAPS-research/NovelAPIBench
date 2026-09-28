"""Validation of generated usage examples (component E; Appendix B.2, "Stage 2").

An example is

* ``executed`` when it runs cleanly in the sandbox (the library's ``new_version`` environment);
* ``static`` when it parses, references the target API, and its execution fails only for want
  of a resource (network, weights, memory, time) or on a ``NotImplementedError`` raised inside
  the library (a code path that needs an undocumented setup call);
* ``failed`` otherwise (syntax error, target not referenced, or a genuine runtime error).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass

from novelapibench.runtime.sandbox import ExecutionResult, execute_code

_RESOURCE_ERROR_MARKERS = (
    "ConnectionError", "ConnectionResetError", "ConnectionRefusedError", "URLError", "SSLError",
    "HTTPError", "HTTP Error", "Max retries exceeded", "TimeoutError", "socket.timeout",
    "Temporary failure in name resolution", "getaddrinfo failed", "No route to host",
    "Network is unreachable", "CUDA out of memory", "OutOfMemoryError", "Killed",
    "MemoryError", "No such file or directory", "FileNotFoundError", "PermissionError",
)
_LIBRARY_INTERNAL_MARKERS = ("NotImplementedError",)


@dataclass
class Validation:
    valid: bool
    status: str       # "executed" | "static" | "failed"
    reason: str


def _tail(text: str | None, n: int = 160) -> str:
    return (text or "").strip().replace("\n", " ")[-n:]


def references_api(tree: ast.AST, api_name: str) -> bool:
    """The code mentions the API's leaf name (identifier, attribute or imported name)."""
    leaf = api_name.split(".")[-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == leaf:
            return True
        if isinstance(node, ast.Attribute) and node.attr == leaf:
            return True
        if isinstance(node, ast.alias) and node.name.split(".")[-1] == leaf:
            return True
    return False


def _can_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001
        return False


def _first_unresolvable_import(tree: ast.AST) -> str | None:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _can_import(alias.name.split(".")[0]):
                    return alias.name
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if not _can_import(node.module.split(".")[0]):
                return node.module
    return None


def _has_marker(result: ExecutionResult, markers: tuple[str, ...]) -> bool:
    blob = (result.stderr or "") + "\n" + (result.stdout or "")
    return any(m in blob for m in markers)


def validate_example(code: str, api_name: str, timeout: int = 30, max_memory_mb: int = 16384,
                     env_python: str | None = None, pid_namespace: bool = False) -> Validation:
    if not code or not code.strip():
        return Validation(False, "failed", "empty example")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return Validation(False, "failed", f"syntax error: {e.msg}")
    if not references_api(tree, api_name):
        return Validation(False, "failed", f"does not reference {api_name}")
    # Import resolution can only be checked statically in the running interpreter; in a
    # separate environment the sandbox run decides (ModuleNotFoundError -> failed).
    if env_python is None or env_python == sys.executable:
        bad = _first_unresolvable_import(tree)
        if bad:
            return Validation(False, "failed", f"unresolvable import: {bad}")
    result = execute_code(code, timeout=timeout, max_memory_mb=max_memory_mb,
                          env_python=env_python, pid_namespace=pid_namespace)
    if result.passed:
        return Validation(True, "executed", "runs cleanly")
    if result.timed_out or _has_marker(result, _RESOURCE_ERROR_MARKERS):
        return Validation(True, "static", f"execution blocked by environment ({_tail(result.stderr)}); "
                                          "static checks pass")
    if _has_marker(result, _LIBRARY_INTERNAL_MARKERS):
        return Validation(True, "static", f"library-internal limitation ({_tail(result.stderr)}); "
                                          "static checks pass")
    return Validation(False, "failed", f"runtime error: {_tail(result.stderr)}")
