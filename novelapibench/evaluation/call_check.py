"""Call-record check (Appendix B.3): the completion's own calls to the target API must
reproduce the reference solution's calls.

The target-call monitor is replaced by a *recording* monitor that, for every call the
completion makes to the target API after ``context_code`` has run, stores a fingerprint of the
return value and of the explicitly passed arguments (bound to the target's signature, taken
after the call so in-place APIs are covered; free-text strings by type only). The reference
solution is run the same way to obtain the expected records. A completion passes the check when
every expected record is matched by one of its own calls:

* for a function call that returns something informative (a value, shape, dtype, checksum,
  length or a non-default repr), the return fingerprint must match;
* otherwise (None, an opaque object, a function such as a decorator) the argument fingerprint
  must match instead;
* for a constructor call, the argument fingerprint must match (the instance's repr is not
  compared: it embeds the arguments' reprs and often omits the configuration).

A call that raises is recorded with the exception type in place of the return value.
Fingerprint fields that vary between reference runs (random inputs, object addresses,
timestamps) are dropped: the reference is run four times, twice with its constant seeds
replaced by random ones and different global seeds (``expected_from_runs``).

The monitor, fingerprint and check are injected as source text, because they run inside the
per-library sandbox interpreter, not in this process.
"""
from __future__ import annotations

import ast
import json
import math

MAX_RECORDS = 200

# ---------------------------------------------------------------------------
# Sandbox-side code (text)
# ---------------------------------------------------------------------------

_FINGERPRINT_CODE = r'''
import hashlib as _ck_hl
import inspect as _ck_inspect
import math as _ck_math
import re as _ck_re
_CK_ADDR = _ck_re.compile(r"0x[0-9a-fA-F]+")
_CK_DEFAULT_REPR = _ck_re.compile(r"^<[\w.<>]+ object at 0x\?>$")


def _ck_h(s):
    return _ck_hl.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


def _ck_num(x):
    try:
        x = float(x)
    except Exception:
        return None
    if _ck_math.isnan(x):
        return "nan"
    if _ck_math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _ck_tname(o):
    t = type(o)
    mod = getattr(t, "__module__", "") or ""
    if isinstance(o, bool):
        return "bool"
    if isinstance(o, (int, float)) or (mod == "numpy" and getattr(o, "shape", None) == () and hasattr(o, "item")):
        return "number"  # a numpy.float64 where the reference passed a float is the same value
    if mod == "__main__":
        return "__main__"  # a class the script defined: its name is the author's choice
    if isinstance(o, dict):
        return "mapping"  # a defaultdict / OrderedDict where the reference passed a dict is the same argument
    return mod.split(".")[0] + "." + getattr(t, "__qualname__", t.__name__)


def _ck_array(o, d):
    try:
        d["shape"] = [int(s) for s in o.shape]
    except Exception:
        pass
    try:
        d["dtype"] = str(o.dtype)
    except Exception:
        pass
    try:
        if hasattr(o, "detach"):
            x = o.detach().cpu()
            if getattr(x, "is_sparse", False):
                x = x.to_dense()
            if x.is_complex():
                x = x.abs()
            x = x.double()
            d["nan"] = int(x.isnan().sum().item())
            d["s"], d["a"] = _ck_num(x.nansum().item()), _ck_num(x.abs().nansum().item())
            return
        import numpy as _ck_np
        x = _ck_np.asarray(o)
        if x.dtype.kind in "biufc":
            x = _ck_np.abs(x) if x.dtype.kind == "c" else x.astype("float64")
            d["nan"] = int(_ck_np.isnan(x).sum())
            d["s"], d["a"] = _ck_num(_ck_np.nansum(x)), _ck_num(_ck_np.nansum(_ck_np.abs(x)))
        else:
            d["h"] = _ck_h(repr(x.tolist())[:50000])
    except Exception:
        pass


def _ck_pandas(o, d):
    try:
        d["shape"] = [int(s) for s in o.shape]
    except Exception:
        pass
    try:
        if hasattr(o, "columns"):
            d["cols"] = _ck_h(repr([str(c) for c in o.columns]))
            d["dtypes"] = _ck_h(repr([str(t) for t in o.dtypes]))
            num = o.select_dtypes("number")
            rest = o.drop(columns=num.columns)
        else:
            d["dtype"] = str(o.dtype)
            num, rest = (o, None) if o.dtype.kind in "biuf" else (None, o)
        if num is not None and getattr(num, "size", 0):
            v = num.to_numpy(dtype="float64", na_value=float("nan"))
            import numpy as _ck_np
            d["s"] = _ck_num(_ck_np.nansum(v))
            d["a"] = _ck_num(_ck_np.nansum(_ck_np.abs(v)))
        if rest is not None and getattr(rest, "size", 0):
            d["h"] = _ck_h(rest.astype(str).to_csv()[:200000])
        if hasattr(o, "index"):
            d["idx"] = _ck_h(repr([str(i) for i in list(o.index)[:2000]]))
    except Exception:
        pass


def _ck_fp(o, _depth=0):
    d = {"t": _ck_tname(o)}
    try:
        if o is None:
            return d
        if isinstance(o, bool):
            d["v"] = bool(o)
        elif isinstance(o, int):
            d["v"] = int(o) if abs(o) < 2 ** 53 else str(o)
        elif isinstance(o, float):
            d["v"] = _ck_num(o)
        elif d["t"] == "number":  # numpy scalar
            k = getattr(getattr(o, "dtype", None), "kind", "")
            d["v"] = _ck_num(abs(o.item())) if k == "c" else int(o.item()) if k in "biu" else _ck_num(o.item())
        elif isinstance(o, complex):
            d["v"] = [_ck_num(o.real), _ck_num(o.imag)]
        elif isinstance(o, str):
            d["len"], d["h"] = len(o), _ck_h(_CK_ADDR.sub("0x?", o))
        elif isinstance(o, (bytes, bytearray)):
            d["len"], d["h"] = len(o), _ck_h(bytes(o)[:100000].hex())
        elif (getattr(type(o), "__module__", "") or "").startswith("pandas") and hasattr(o, "shape"):
            _ck_pandas(o, d)
        elif hasattr(o, "shape") and hasattr(o, "dtype"):
            _ck_array(o, d)
        elif isinstance(o, (list, tuple)):
            d["t"] = "sequence"  # a list where the reference passed a tuple is the same argument
            d["len"] = len(o)
            if _depth < 3:
                d["items"] = [_ck_fp(x, _depth + 1) for x in list(o)[:8]]
        elif isinstance(o, (set, frozenset)):
            d["len"] = len(o)
            d["h"] = _ck_h(repr(sorted(_CK_ADDR.sub("0x?", repr(x)) for x in o))[:50000])
        elif isinstance(o, dict):
            d["len"] = len(o)
            ks = sorted(o, key=lambda k: repr(k))
            d["keys"] = _ck_h(repr([repr(k) for k in ks]))
            if _depth < 3:
                d["items"] = [_ck_fp(o[k], _depth + 1) for k in ks[:8]]
        elif _ck_inspect.isroutine(o) or type(o).__name__ == "partial" or _ck_inspect.iscoroutine(o) \
                or _ck_inspect.isgenerator(o) or _ck_inspect.isasyncgen(o):
            # functions, coroutines, generators: only the type. Their names are the author's
            # choice (a decorated function, an async helper), not API behaviour.
            pass
        elif isinstance(o, type) and getattr(o, "__module__", "") == "__main__":
            pass  # a class the script defined: its name is the author's choice
        else:
            try:
                r = _CK_ADDR.sub("0x?", repr(o))
                if not _CK_DEFAULT_REPR.match(r):
                    d["r"] = _ck_h(r[:50000])
            except Exception:
                pass
            if hasattr(o, "__len__") and not isinstance(o, type):
                try:
                    d["len"] = len(o)
                except Exception:
                    pass
    except Exception:
        pass
    return d


def _ck_argfp(v):
    # Free text (a docstring, a description, a label: any string with whitespace) is the
    # author's choice, not API usage: only its type is kept. Tokens ('gelu', a file name) are
    # compared in full.
    d = _ck_fp(v)
    if isinstance(v, str) and any(c.isspace() for c in v):
        return {"t": d["t"]}
    return d


def _ck_canon(fn, a, kw):
    """Fingerprints of the arguments the caller passed explicitly, bound to the signature.

    Every bound argument is recorded (defaults applied) and the explicitly passed ones are listed
    under ``__explicit__``. The expected record keeps only the reference's explicit arguments
    (`expected_from_runs`): an argument left at its default is not API usage the reference
    demonstrates, and a completion that adds an extra keyword is not reproducing the reference
    less. A completion that omits an explicit argument is compared on the default it then gets
    (`radius=None` written out or left out is the same call); another value differs.
    """
    try:
        b = _ck_inspect.signature(fn).bind(*a, **kw)
        explicit = set(b.arguments)
        b.apply_defaults()
        out = {"__explicit__": []}
        for k, v in b.arguments.items():
            p = b.signature.parameters[k]
            if p.kind == p.VAR_POSITIONAL:
                out[k] = [_ck_argfp(x) for x in v]
                names = [k]
            elif p.kind == p.VAR_KEYWORD:
                names = []
                for kk, vv in sorted(v.items()):
                    out[kk] = _ck_argfp(vv)
                    names.append(kk)
            else:
                out[k] = _ck_argfp(v)
                names = [k]
            if k in explicit:
                out["__explicit__"] += names
        return out
    except Exception:
        return {"*": [_ck_argfp(x) for x in a], "**": {k: _ck_argfp(v) for k, v in sorted(kw.items())}}
'''

_RECORDING_SPY_TEMPLATE = """\
# --- target-API call spy with call records (call-record check, auto-injected) ---
import importlib as _spy_il
import sys as _spy_sys
{fingerprint}
_target_api_name = {api_name!r}
_target_api_call_count = [0]
_target_api_calls = []
_target_api_recording = [False]
_target_api_depth = [0]
_target_api_spy_installed = False


def _ck_record(rec):
    if _target_api_recording[0] and len(_target_api_calls) < {max_records}:
        _target_api_calls.append(rec)


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
            class _target_api_spy(_orig):
                def __init__(self, *_a, **_kw):
                    _target_api_call_count[0] += 1
                    _target_api_depth[0] += 1
                    try:
                        super().__init__(*_a, **_kw)
                    except BaseException as _e:
                        _target_api_depth[0] -= 1
                        if _target_api_depth[0] == 0:
                            _ck_record({{"kind": "init", "ret": {{"t": "raised", "exc": type(_e).__name__}},
                                         "args": _ck_canon(_orig, _a, _kw)}})
                        raise
                    _target_api_depth[0] -= 1
                    if _target_api_depth[0] == 0:
                        _sub = type(self) is not _target_api_spy
                        _ck_record({{"kind": "init", "ret": {{"t": "instance"}} if _sub else _ck_fp(self),
                                     "args": _ck_canon(_orig, _a, _kw), "subclassed": _sub}})
                def __init_subclass__(cls, **_kw):
                    _target_api_call_count[0] += 1
                    super().__init_subclass__(**_kw)
                    _ck_record({{"kind": "subclass"}})
        else:
            def _target_api_spy(*_a, **_kw):
                _target_api_call_count[0] += 1
                _target_api_depth[0] += 1
                try:
                    _ret = _orig(*_a, **_kw)
                except BaseException as _e:
                    _target_api_depth[0] -= 1
                    if _target_api_depth[0] == 0:
                        _ck_record({{"kind": "call", "ret": {{"t": "raised", "exc": type(_e).__name__}},
                                     "args": _ck_canon(_orig, _a, _kw)}})
                    raise
                _target_api_depth[0] -= 1
                if _target_api_depth[0] == 0:
                    _ck_record({{"kind": "call", "ret": _ck_fp(_ret), "args": _ck_canon(_orig, _a, _kw)}})
                return _ret
            try:
                _target_api_spy.__wrapped__ = _orig
            except Exception:
                pass
        _top = _parts[0]
        _orig_mod = getattr(_orig, "__module__", None)
        for _mn, _m in list(_spy_sys.modules.items()):
            if _m is None:
                continue
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

#: Runs after context_code (and the leaf binding): from here on, target calls are the completion's.
START_RECORDING = """\
try:
    del _target_api_calls[:]
    _target_api_recording[0] = True
except NameError:
    pass
"""

_CHECK_TEMPLATE = """\
# --- call-record check (auto-injected) ---
_target_api_recording[0] = False
_ck_expected = {expected}


def _ck_match(e, a):
    if isinstance(e, dict):
        return isinstance(a, dict) and all(k in a and _ck_match(v, a[k]) for k, v in e.items())
    if isinstance(e, list):
        return isinstance(a, list) and len(e) == len(a) and all(_ck_match(x, y) for x, y in zip(e, a))
    if isinstance(e, float) and isinstance(a, (int, float)) and not isinstance(a, bool):
        return _ck_math.isclose(e, a, rel_tol=1e-6, abs_tol=1e-9)
    return e == a


for _ck_i, _ck_e in enumerate(_ck_expected):
    assert any(_ck_match(_ck_e, _ck_a) for _ck_a in _target_api_calls), \\
        f"no call to the target API reproduces reference call {{_ck_i + 1}}/{{len(_ck_expected)}} " \\
        f"({{'return value' if 'args' not in _ck_e else 'arguments'}} differ)"
# --- end call-record check ---
"""

def dump_records(path: str) -> str:
    """Test code that writes the recorded calls to `path` (the sandbox keeps only 4 KB of stdout)."""
    return (
        "_target_api_recording[0] = False\n"
        "import json as _ck_json\n"
        f"with open({path!r}, 'w') as _ck_f:\n"
        "    _ck_json.dump(_target_api_calls, _ck_f)\n"
    )


def build_recording_spy(api_name: str) -> str:
    return _RECORDING_SPY_TEMPLATE.format(fingerprint=_FINGERPRINT_CODE, api_name=api_name,
                                          max_records=MAX_RECORDS)


def build_call_check(expected: list[dict]) -> str:
    return _CHECK_TEMPLATE.format(expected=repr(expected))


def read_records(path: str) -> list[dict] | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Expected records from several reference runs
# ---------------------------------------------------------------------------

SEED_FUNCS = {"seed", "manual_seed", "manual_seed_all", "set_seed", "seed_everything",
              "default_rng", "RandomState", "Random", "PRNGKey", "Generator"}
SEED_KWARGS = {"seed", "random_state", "random_seed", "rng_seed"}
_RANDOM_INT = "__import__('random').randrange(1, 2 ** 31)"


class _Unseed(ast.NodeTransformer):
    """Replace constant seeds with fresh random ones, so seeded randomness shows up as variation."""

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
        if name in SEED_FUNCS and node.args and isinstance(node.args[0], ast.Constant):
            node.args[0] = ast.parse(_RANDOM_INT, mode="eval").body
        for kw in node.keywords:
            if kw.arg in SEED_KWARGS and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                kw.value = ast.parse(_RANDOM_INT, mode="eval").body
        return node


def unseed(code: str) -> str | None:
    """`code` with constant seeds randomised, or None when it does not parse on its own."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return ast.unparse(ast.fix_missing_locations(_Unseed().visit(tree)))


def reseed_prelude(seed: int, uses_torch: bool = False) -> str:
    """Global reseed before the completion.

    numpy and `random` seed themselves from OS entropy, but torch's default generator starts
    from a fixed seed, so unseeded torch randomness would look deterministic across runs. torch
    is reseeded when it is already imported or the task's code mentions it (`uses_torch`);
    importing it unconditionally costs seconds per run in every env that ships it.
    """
    return (
        "import random as _ck_random\n"
        "import sys as _ck_sys\n"
        f"_ck_random.seed({seed})\n"
        "if 'numpy' in _ck_sys.modules:\n"
        f"    _ck_sys.modules['numpy'].random.seed({seed})\n"
        f"if 'torch' in _ck_sys.modules or {bool(uses_torch)}:\n"
        "    import torch as _ck_torch0\n"
        f"    _ck_torch0.manual_seed({seed})\n"
    )


def _close(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    return a == b


def stable(a, b):
    """The part of fingerprint `a` that `b` reproduces; None when nothing is shared."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for k, v in a.items():
            if k in b:
                s = stable(v, b[k])
                if s is not None:
                    out[k] = s
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        parts = [stable(x, y) for x, y in zip(a, b)]
        return [p if p is not None else {} for p in parts]
    return a if _close(a, b) else None


_INFORMATIVE = {"v", "len", "shape", "dtype", "s", "a", "h", "r", "items", "keys", "cols", "dtypes", "idx", "exc"}


def informative(fp: dict) -> bool:
    return isinstance(fp, dict) and bool(_INFORMATIVE & set(fp))


def expected_from_runs(runs: list[list[dict]]) -> tuple[list[dict], str]:
    """Expected records from several reference runs, and a status.

    Status: ``ok``; ``unstable`` when the runs disagree on how many calls were made (then only
    records identical in kind are kept, compared by their shared fields, first-run order);
    ``no_calls`` when the reference made no recorded call (e.g. all its target calls are nested
    inside another target call, or happen only while context_code runs).
    """
    first = runs[0]
    if not first:
        return [], "no_calls"
    status = "ok" if all(len(r) == len(first) for r in runs) else "unstable"
    exp = []
    for i, rec in enumerate(first):
        cur = rec
        for other in runs[1:]:
            if status == "ok":
                cur = stable(cur, other[i]) or {}
            else:
                cand = [stable(cur, o) or {} for o in other if o.get("kind") == rec.get("kind")]
                cur = max(cand, key=lambda c: len(json.dumps(c))) if cand else {}
        kind = rec.get("kind")
        ret = cur.get("ret", {})
        args = cur.get("args", {})
        if "__explicit__" in rec.get("args", {}):
            keep = set(rec["args"]["__explicit__"])
            args = {k: v for k, v in args.items() if k in keep}
        cur = {**cur, "args": args}
        if kind == "subclass":
            e = {"kind": "subclass"}
        elif kind == "init":
            # a constructor's arguments are the API usage. The instance's repr is not compared:
            # it embeds the arguments' reprs (a defaultdict prints differently from an equal dict)
            # and often omits the configuration (an nn.Module without extra_repr).
            e = {"kind": kind, "args": cur.get("args", {})}
        elif informative(ret):
            e = {"kind": kind, "ret": ret}
        else:
            e = {"kind": kind, "args": cur.get("args", {})}
        exp.append(e)
    seen, out = set(), []
    for e in exp:
        k = json.dumps(e, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out, status


# ---------------------------------------------------------------------------
# Harness assembly
# ---------------------------------------------------------------------------

SPY_CHECK_END = "# --- end check ---"


def build_setup(api_name: str, setup_code: str | None, context_code: str | None,
                bind_target_leaf: bool = True, prelude: str = "") -> str:
    """:func:`harness.build_setup` with the recording monitor in place of the task's monitor, and
    recording switched on once ``context_code`` has run."""
    from novelapibench.evaluation.harness import (
        POST_CONTEXT_MONITOR_RESET, build_monitor_setup, leaf_binding_for_api, module_import_for_api)
    if (setup_code or "").strip() != build_monitor_setup(api_name).strip():
        raise ValueError(f"setup_code of {api_name} is not the stock target-call monitor; the "
                         "call-record check would drop part of it")
    return "\n\n".join(filter(None, [
        module_import_for_api(api_name),
        build_recording_spy(api_name),
        context_code or "",
        leaf_binding_for_api(api_name) if bind_target_leaf else "",
        POST_CONTEXT_MONITOR_RESET,
        START_RECORDING,
        prelude,
    ]))


def build_test(execution_test: str, expected: list[dict]) -> str:
    """The task's test with the call-record check right after its target-call assertion."""
    head, sep, tail = execution_test.partition(SPY_CHECK_END)
    if not sep:
        raise ValueError("execution_test has no target-call check block")
    return head + sep + "\n" + build_call_check(expected) + tail
