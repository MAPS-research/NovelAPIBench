"""Execution harness for a task (Appendix B.3).

A completion is executed as one script::

    import <target module>              # so pkg.mod.Sym(...) resolves
    <setup_code>                        # target-call monitor (installed by construction)
    <context_code>
    <leaf binding>                      # so a bare Sym(...) resolves the same way
    <monitor reset>                     # calls made by context_code do not count
    <completion>
    <execution_test>                    # monitor assertion, [call-record check], scenario asserts

The monitor wraps the target API in every module that re-exports it, so a completion passes
only if it actually calls the target. Construction (Stage 3/4) builds tasks with this harness
without the leaf binding; evaluation adds the binding and the call-record check.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Target-call monitor (text injected into the sandboxed script)
# ---------------------------------------------------------------------------

MONITOR_SETUP_TEMPLATE = """\
# --- target-API call spy (auto-injected) ---
import importlib as _spy_il
import sys as _spy_sys
_target_api_name = {api_name!r}
_target_api_call_count = [0]
_target_api_spy_installed = False
try:
    _parts = _target_api_name.split(".")
    _mod = None
    _split = 0
    for _i in range(len(_parts), 0, -1):
        try:
            _mod = _spy_il.import_module(".".join(_parts[:_i]))
            _split = _i
            break
        except ImportError:
            continue
    if _mod is not None:
        _parent = _mod
        for _a in _parts[_split:-1]:
            _parent = getattr(_parent, _a)
        _leaf = _parts[-1]
        _orig = getattr(_parent, _leaf)
        if isinstance(_orig, type):
            # Class: subclass so `class X(Orig): ...` still works and counts.
            class _target_api_spy(_orig):
                def __init__(self, *_a, **_kw):
                    _target_api_call_count[0] += 1
                    super().__init__(*_a, **_kw)
                def __init_subclass__(cls, **_kw):
                    _target_api_call_count[0] += 1
                    super().__init_subclass__(**_kw)
        else:
            def _target_api_spy(*_a, **_kw):
                _target_api_call_count[0] += 1
                return _orig(*_a, **_kw)
            try:
                _target_api_spy.__wrapped__ = _orig
            except Exception:
                pass
        # Patch every re-export: any module attribute in sys.modules that is
        # _orig itself. Covers `from pkg.sub import X` -> `pkg.X` aliases.
        _top = _parts[0]
        _orig_mod = getattr(_orig, "__module__", None)
        for _mn, _m in list(_spy_sys.modules.items()):
            if _m is None:
                continue
            # Only scan the target's own top-level package + its declaring
            # module's top-level (covers typing.cast re-exported in fastapi).
            _mtop = _mn.split(".", 1)[0]
            if _mtop != _top and (not _orig_mod or _mtop != _orig_mod.split(".", 1)[0]):
                continue
            try:
                _names = list(vars(_m))
            except Exception:
                continue
            for _n in _names:
                try:
                    if getattr(_m, _n, None) is _orig:
                        setattr(_m, _n, _target_api_spy)
                except Exception:
                    continue
        # Belt-and-suspenders: ensure the canonical FQN is patched.
        try:
            if getattr(_parent, _leaf, None) is _orig:
                setattr(_parent, _leaf, _target_api_spy)
        except Exception:
            pass
        _target_api_spy_installed = True
except Exception:
    pass
# --- end spy setup ---
"""

MONITOR_CHECK_TEMPLATE = """\
# --- target-API call check (auto-injected) ---
assert _target_api_spy_installed, \\
    "target-API spy failed to install for {api_name}"
assert _target_api_call_count[0] > 0, \\
    "solution did not call target API {api_name}"
_target_api_call_count[0] = 0
# --- end check ---
"""



def build_monitor_setup(api_name: str) -> str:
    """The ``setup_code`` every task carries: installs the counting target-call monitor."""
    return MONITOR_SETUP_TEMPLATE.format(api_name=api_name)


def build_monitor_check(api_name: str) -> str:
    """The assertion that opens every ``execution_test``: the target was called."""
    return MONITOR_CHECK_TEMPLATE.format(api_name=api_name)


def module_import_for_api(api_name: str) -> str:
    """``import package.module`` for a dotted API name."""
    parts = api_name.rsplit(".", 1)
    if len(parts) < 2:
        return ""
    return f"import {parts[0]}"


def leaf_binding_for_api(api_name: str) -> str:
    """Bind the target's leaf symbol when nothing earlier in the script did.

    ``module_import_for_api`` lets ``pkg.mod.Sym(...)`` resolve without the completion writing
    an import; this binding lets a bare ``Sym(...)`` resolve the same way, so the two import
    styles are scored alike. It is skipped when the name already resolves (so it never replaces
    an existing object), and it runs after the monitor is installed, so the bound name is the
    monitoring wrapper: a completion still passes only by calling the target API.
    """
    mod, _, leaf = api_name.rpartition(".")
    if not mod or not leaf.isidentifier():
        return ""
    return (f"try:\n    {leaf}\nexcept NameError:\n    try:\n"
            f"        from {mod} import {leaf}\n    except Exception:\n        pass\n")


POST_CONTEXT_MONITOR_RESET = """\
# Reset spy counter AFTER context_code — context may indirectly call the
# target API as a side effect (e.g. MDAnalysis.Universe.empty() internally
# calls get_guesser). Without this reset, test-body `count > 0` assertions
# can pass even when the candidate solution is a no-op. Guarded against
# NameError so non-spy harnesses / llm_only paths are unaffected.
try:
    _target_api_call_count[0] = 0
except NameError:
    pass
"""



def build_setup(api_name: str, setup_code: str | None, context_code: str | None,
                bind_target_leaf: bool = False) -> str:
    """Setup part of the flat execution script (everything before the completion).

    ``bind_target_leaf`` is off during construction (reference validation, C2, C3), as when the
    benchmark was built, and on during evaluation.
    """
    return "\n\n".join(filter(None, [
        module_import_for_api(api_name),
        setup_code,
        context_code or "",
        leaf_binding_for_api(api_name) if bind_target_leaf else "",
        POST_CONTEXT_MONITOR_RESET,
    ]))
