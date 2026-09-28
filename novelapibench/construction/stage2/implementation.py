"""Implementation code C (Appendix B.2, "Stage 2").

C is the API's source (the whole class body for a class) followed by the same-module functions
and classes it references, one level deep, with every docstring removed (natural-language
descriptions belong to M). Helpers from other modules (numpy, torch, ...) are not included.
Two shapes carry nothing API-specific and yield no C: a factory-generated closure wrapper
(``functools.wraps`` wrappers, single-``return`` dispatchers, functions named ``wrapper``...)
and a module-level ``name = lambda ...`` alias.

Extraction runs inside the library's environment: this module is standard-library only, and
its own source is sent to that interpreter together with a small driver.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path


def strip_all_docstrings(source: str) -> str:
    """Remove the docstring of every module, class and function (via ``ast.unparse``)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    class _Cleaner(ast.NodeTransformer):
        def _drop(self, node):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] if len(body) > 1 else [ast.Pass()]
            return node

        def visit_Module(self, node):
            self.generic_visit(node)
            return self._drop(node)

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            return self._drop(node)

        def visit_AsyncFunctionDef(self, node):
            self.generic_visit(node)
            return self._drop(node)

        def visit_ClassDef(self, node):
            self.generic_visit(node)
            return self._drop(node)

    tree = _Cleaner().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:  # noqa: BLE001
        return source


def _resolve_api(api_name: str):
    """``(object, defining module)`` for a dotted name (longest importable prefix first)."""
    parts = api_name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001
            continue
        obj = mod
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        owning = getattr(obj, "__module__", None)
        if owning:
            try:
                return obj, importlib.import_module(owning)
            except Exception:  # noqa: BLE001
                pass
        return obj, mod
    return None, None


def _referenced_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
    return names


def _same_module_helpers(obj, owning_module, target_source: str) -> list[tuple[str, object]]:
    mod_globals = vars(owning_module)
    own_name = getattr(obj, "__name__", None)
    helpers, seen = [], set()
    for name in sorted(_referenced_names(target_source)):
        if name == own_name:
            continue
        val = mod_globals.get(name)
        if val is None or not (inspect.isfunction(val) or inspect.isclass(val)):
            continue
        if getattr(val, "__module__", None) != owning_module.__name__ or id(val) in seen:
            continue
        seen.add(id(val))
        helpers.append((name, val))
    return helpers


_GENERIC_WRAPPER_NAMES = frozenset({"wrapper", "inner", "wrapped", "_wrapper", "_inner", "_wrapped"})


def _has_wraps_decorator(fn_node) -> bool:
    for dec in fn_node.decorator_list:
        if isinstance(dec, ast.Call):
            dec = dec.func
        name = dec.attr if isinstance(dec, ast.Attribute) else dec.id if isinstance(dec, ast.Name) else None
        if name == "wraps":
            return True
    return False


def _is_factory_generated_closure(obj) -> bool:
    """A closure returned by a factory that only dispatches to another callable."""
    if not inspect.isfunction(obj):
        return False
    code = getattr(obj, "__code__", None)
    if code is None or not getattr(code, "co_freevars", ()):
        return False
    if getattr(obj, "__name__", "") in _GENERIC_WRAPPER_NAMES:
        return True
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _has_wraps_decorator(node):
                return True
            body = [s for s in node.body if not (isinstance(s, ast.Expr)
                                                 and isinstance(s.value, ast.Constant)
                                                 and isinstance(s.value.value, str))]
            return len(body) == 1 and isinstance(body[0], ast.Return)
    return False


def _is_lambda_alias(source: str) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    return (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
            and isinstance(tree.body[0].value, ast.Lambda))


def extract_implementation_inproc(api_name: str) -> str | None:
    """C for an API importable in the running interpreter, or None."""
    obj, owning = _resolve_api(api_name)
    if obj is None or owning is None or _is_factory_generated_closure(obj):
        return None
    try:
        target_source = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        return None
    if _is_lambda_alias(target_source):
        return None
    parts = [strip_all_docstrings(target_source)]
    for name, helper in _same_module_helpers(obj, owning, target_source):
        try:
            helper_source = textwrap.dedent(inspect.getsource(helper))
        except (OSError, TypeError):
            continue
        parts.append(f"# --- helper: {owning.__name__}.{name} ---\n"
                     f"{strip_all_docstrings(helper_source)}")
    return "\n\n".join(parts)


_OUTPUT_SENTINEL = "__IMPLEMENTATION_OUTPUT__"


def extract_implementation(api_name: str, env_python: str | None = None,
                           timeout: int = 30) -> str | None:
    """C for ``api_name``, extracted inside ``env_python`` when that is another interpreter.

    Returns None on any failure (not importable, no Python source, timeout).
    """
    if env_python is None or Path(env_python) == Path(sys.executable):
        return extract_implementation_inproc(api_name)
    driver = textwrap.dedent(f"""
        import json, sys
        try:
            code = extract_implementation_inproc({api_name!r})
        except Exception:
            code = None
        sys.stdout.write({_OUTPUT_SENTINEL!r} + json.dumps({{"code": code}}))
        """)
    script = Path(__file__).read_text(encoding="utf-8") + "\n" + driver
    try:
        proc = subprocess.run([env_python, "-c", script], capture_output=True, text=True,
                              timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    idx = proc.stdout.rfind(_OUTPUT_SENTINEL)
    if idx < 0:
        return None
    try:
        return json.loads(proc.stdout[idx + len(_OUTPUT_SENTINEL):].strip()).get("code")
    except json.JSONDecodeError:
        return None
