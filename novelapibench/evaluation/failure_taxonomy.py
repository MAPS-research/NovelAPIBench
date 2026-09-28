"""Failure diagnosis (paper Section 3.3, Appendix C).

Every failed completion receives one of six mutually exclusive labels:

    WrongAPISelection  chose or hallucinated a different API, or never called the target
    WrongImport        referred to the target through a wrong or missing module path
    WrongSyntax        syntactically invalid code (or an empty stub)
    WrongParam         the target API's own signature rejected the arguments
    WrongShapeDtype    the target API rejected the structure of its input data
    WrongLogic         the target was called acceptably; the surrounding program is wrong
                       (a crash elsewhere, a wrong callback contract, wrongly prepared inputs,
                       a call that does not reproduce the reference call record)

Rules first: syntax/indentation errors -> WrongSyntax; a completion that finishes without
calling the target (the monitor assertion) -> WrongAPISelection; a ``NameError`` on the target's
leaf name or top-level package -> WrongImport. Everything else goes to GPT-5-mini, which picks
one of four groups (W_API, W_PARAM, W_SHAPE_DTYPE, W_LOGIC; the prompt is the paper's
classifier listing); a W_API answer is then mapped to one of the three API-level labels from the
exception type and the rationale.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from novelapibench.llm.strong import StrongLLM
from novelapibench.runtime.sandbox import ExecutionResult
from novelapibench.schemas import Task

PASS = "Pass"
LABELS = ("WrongAPISelection", "WrongImport", "WrongSyntax",
          "WrongParam", "WrongShapeDtype", "WrongLogic")

#: Groups the LLM judge chooses from (the vocabulary of the classifier prompt).
JUDGE_GROUPS = ("W_API", "W_PARAM", "W_SHAPE_DTYPE", "W_LOGIC")
_API_SUBGROUP_LABEL = {"W_API_SELECT": "WrongAPISelection", "W_API_IMPORT": "WrongImport",
                       "W_API_CODE": "WrongSyntax"}
_GROUP_LABEL = {"W_PARAM": "WrongParam", "W_SHAPE_DTYPE": "WrongShapeDtype", "W_LOGIC": "WrongLogic"}


def to_label(group: str, api_subgroup: str = "") -> str:
    """Paper label for a judge group (and, for W_API, its deterministic sub-group)."""
    if group == "W_API":
        return _API_SUBGROUP_LABEL.get(api_subgroup, "WrongAPISelection")
    return _GROUP_LABEL[group]


@dataclass
class _ASTExtract:
    api_called_name: str | None
    kwargs_extracted: dict[str, str]


def extract_api_call(code: str, target_api: str) -> _ASTExtract:
    """Find the first call to the target API's leaf name and extract its kwargs.

    Lenient: returns whatever leaf name is actually called (may differ from
    target_api when the model picked the wrong API). If no matching call is
    found, ``api_called_name`` is the leaf name of the first ast.Call found,
    or ``None`` if the code has no calls at all.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _ASTExtract(None, {})

    target_leaf = target_api.split(".")[-1] if target_api else None
    first_call_leaf: str | None = None
    target_match: ast.Call | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        leaf = _call_leaf(node)
        if leaf is None:
            continue
        if first_call_leaf is None:
            first_call_leaf = leaf
        if target_leaf and leaf == target_leaf and target_match is None:
            target_match = node

    if target_match is not None:
        return _ASTExtract(
            api_called_name=target_leaf,
            kwargs_extracted=_unparse_kwargs(target_match),
        )
    return _ASTExtract(api_called_name=first_call_leaf, kwargs_extracted={})


def _call_leaf(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _unparse_kwargs(node: ast.Call) -> dict[str, str]:
    out: dict[str, str] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            out[kw.arg] = ast.unparse(kw.value)
        except Exception:
            out[kw.arg] = "<unrepr>"
    return out


def extract_exception_type(stderr: str) -> str:
    """Pull the final exception class name from a traceback, if present."""
    if not stderr:
        return ""
    # Find the last "XxxError: ..." line (traceback summary).
    matches = re.findall(r"^([A-Z][A-Za-z_]*(?:Error|Exception|Warning|Interrupt))\b",
                        stderr, flags=re.MULTILINE)
    return matches[-1] if matches else ""


_NAMEERROR_RE = re.compile(r"NameError: name '([^']+)' is not defined")


def extract_undefined_name(stderr: str) -> str | None:
    """Pull `X` from a `NameError: name 'X' is not defined` traceback line."""
    if not stderr:
        return None
    m = _NAMEERROR_RE.search(stderr)
    return m.group(1) if m else None


_CLASSIFIER_SYSTEM = (
    "You classify a single failed code generation into exactly ONE category. "
    "Output only compact JSON: {\"class\": <CLASS>, \"rationale\": <one sentence>}."
)

_CLASSIFIER_TEMPLATE = """\
Task description:
{description}

Target API: {target_api}
Reference call (from reference solution): {reference_call}

Predicted code (pass@1 sample):
```python
{predicted_code}
```

AST-extracted: api_called_leaf={api_called_leaf}, kwargs={kwargs}
Exception type from execution: {exception_type}
Undefined symbol from stderr: {undefined_name}
Stderr (last 500 chars):
{stderr_snippet}

Pick EXACTLY ONE of: W_API, W_PARAM, W_SHAPE_DTYPE, W_LOGIC.
This prompt is library-agnostic (ML, web frameworks, ORMs, data tooling, etc.) —
do NOT treat any symbol as "commonly known" and therefore excusable. Forgetting
to import a symbol IS a model error; writing the solution around it is part of
the task.

Apply these rules IN ORDER and pick the FIRST that matches.

1) W_API — the model got the API NAME wrong, failed to name it at all, or failed
   to emit code that could invoke it:
   - No call to the target API's leaf name appears anywhere in the predicted code
   - api_called_leaf differs from the target leaf (wrong function called entirely)
   - api_called_leaf matches by NAME but the deepest user-code frame in the
     traceback is in a DIFFERENT top-level package than the target API's package
     (e.g. target_api=scipy.stats.quantile but traceback points into
     '/numpy/lib/function_base.py' → model imported np.quantile instead). The
     surface TypeError may look like W_PARAM; the file-path signal overrides it.
   - An attribute / enum member that does not exist is referenced (AttributeError on a made-up member)
   - NameError ONLY when the undefined symbol is the target API leaf, the
     target API's top-level package, or an obvious alias of the target API.
     Generic scaffolding names like kwargs/output/obj/tmp are NOT enough on
     their own to make this W_API.
   - ModuleNotFoundError — model referenced a module that isn't importable
   - SyntaxError / IndentationError — model failed to emit a valid call
   - Empty stub (pass-only body, comments only) — model gave up on naming the API

2) W_PARAM — right API; the TARGET API's OWN signature rejected the arguments:
   - TypeError raised at the target API's call site: unexpected keyword argument,
     missing required argument, wrong positional arity
   - Required kwarg missing, kwarg name misspelled or invented
   - Literal passed to the target API is the wrong primitive type — str where
     tuple expected (e.g. molSize="(400,300)"), str where int expected, wrong
     enum member, or None where an object was expected because the model
     misread an in-place function's return (e.g. `mol = Chem.SanitizeMol(mol)`
     stores None into mol, then passes None to the target API)
   CAUTION: a TypeError with "unexpected keyword argument X" inside a
   user-supplied CALLBACK (library called model's callback with X; callback
   didn't accept it) is NOT W_PARAM — that's W_LOGIC (callback contract).

3) W_SHAPE_DTYPE — right API, right kwarg names; failure is a NUMERIC /
   STRUCTURAL array-like contract violation raised INSIDE the target API (or
   one layer into its implementation) while validating input data:
   - Tensor / ndarray rank / shape / dtype mismatch (broadcast incompatibility,
     conv kernel vs input mismatch, fromarray dtype rejection, shape unpacking
     on an input array done by the API itself)
   - Dataframe schema / column-dtype / column-count mismatch raised inside the
     target API
   - Response-model / response-schema validation rejection by the target API
   The exception MUST originate inside the target API's body on a check of
   input-array structure. If the exception happens AFTER the target API
   returned — on a subsequent line of the predicted code, or inside a
   model-supplied callback the library invoked — that is W_LOGIC, not
   W_SHAPE_DTYPE.

4) W_LOGIC — right API, plausible args, surrounding program is wrong:
   - AssertionError where the assertion checks semantic correctness (not shape/type)
   - Callback / hook / dependency contract violation — the model wrote a callable
     (e.g. mapping fn for geometric_transform, getForceField for
     ConstrainedEmbed, vectorized fn for nsum, simMetric for SpreadPicker)
     whose signature or shape assumption doesn't match what the library passes
     it. TypeError "unexpected kwarg X" raised INSIDE the model's callback,
     ValueError "too many values to unpack" when the callback tries to
     destructure coords the library handed it, "truth value of array ambiguous"
     from a scalar-only branch in the callback — all W_LOGIC.
   - Wrong control flow: infinite recursion, wrong output variable returned,
     wrong HTTP status
   - Timeout caused by a runaway loop in model-written code wrapping a valid
     API call
   - Return-value misuse: target API was called and returned cleanly, but the
     code on a SUBSEQUENT line misinterprets its return type — treats bytes
     as an object with .save(), f.write(tuple), .skewness() on an ndarray,
     indexes a DataFrame with a set, unpacks a tuple into the wrong number of
     names. The exception is an AttributeError / TypeError / ValueError on the
     line AFTER the target API, not inside it.
   - FileNotFoundError / missing-dependency errors for files the model
     hallucinated in its own scaffolding (e.g. np.load('ground_truth.npy'))
   - Uncaught side-effect error after the target API was successfully invoked

Exclusivity reminders:
- W_SHAPE_DTYPE is ONLY when the target API's own body rejects input shape/
  dtype/rank. Error before the call → W_PARAM. Error after the call returned
  or inside a model-written callback → W_LOGIC.
- "TypeError: unexpected keyword argument" inside a callback the model wrote
  is W_LOGIC, not W_PARAM.
- Same-leaf-different-package (np.quantile vs scipy.stats.quantile) is W_API
  when the traceback file path proves the wrong package was called, even if
  the surface exception looks like W_PARAM.
- NameError is W_API only for target-related undefined symbols; generic
  undefined locals / temporaries should usually fall through to W_LOGIC.
- ModuleNotFoundError / SyntaxError → W_API.
- An empty `def solution(): pass` is W_API.

Output only: {{"class": <CLASS>, "rationale": <one sentence citing the specific signal>}}"""


def _format_reference_call(task: Task) -> str:
    """Return a one-line summary of how the reference solution calls the target API."""
    ref = task.reference_solution or ""
    target_api = task.test_harness.target_api
    leaf = target_api.split(".")[-1] if target_api else ""
    if not ref or not leaf:
        return "<unknown>"
    try:
        tree = ast.parse(ref)
    except SyntaxError:
        return "<unparseable>"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_leaf(node) == leaf:
            try:
                return ast.unparse(node)[:240]
            except Exception:
                return leaf + "(...)"
    return leaf + "(...)"


def _subclassify_w_api(
    error_class: str,
    exception_type: str,
    api_called: bool,
    rationale: str,
    target_api: str,
    api_called_name: str | None = None,
) -> str:
    """Deterministic sub-classification for W_API failures.

    Returns one of W_API_SELECT, W_API_IMPORT, W_API_CODE, or empty string
    when error_class is not W_API.
    """
    if error_class != "W_API":
        return ""

    exc = exception_type or ""
    rationale_lower = (rationale or "").lower()
    target_leaf = target_api.split(".")[-1].lower() if target_api else ""

    # 1. Code-quality failures: syntax errors or empty stubs.
    if exc in ("SyntaxError", "IndentationError"):
        return "W_API_CODE"
    if any(phrase in rationale_lower for phrase in ("empty stub", "pass-only", "pass only")):
        return "W_API_CODE"

    # 2. Import failures: the model referenced the target API in the AST but
    # the runtime could not resolve it (NameError / ImportError / ModuleNotFoundError).
    if exc in ("NameError", "ImportError", "ModuleNotFoundError"):
        if api_called:
            return "W_API_IMPORT"
        if target_leaf and (target_leaf in rationale_lower or "target api symbol" in rationale_lower):
            return "W_API_IMPORT"
        return "W_API_SELECT"

    # 3. api_called=True but assertion failed → spy caught wrong FQN.
    if api_called:
        if exc == "AssertionError":
            return "W_API_SELECT"
        # AttributeError on the target API itself means the model tried to use
        # the correct leaf name on the wrong object / path — still a selection
        # error rather than an import failure.
        if exc == "AttributeError" and target_leaf and target_leaf in rationale_lower:
            return "W_API_SELECT"
        return "W_API_IMPORT"

    # 4. Default: model called a wrong API (same-library, cross-library, or hallucinated).
    return "W_API_SELECT"


# Message of the monitor assertion at the top of every execution test
# (``harness.MONITOR_CHECK_TEMPLATE``).
_SPY_ASSERTION_RE = re.compile(
    r"AssertionError:\s*solution did not call target API", re.IGNORECASE
)


def classify_failure(task: Task, predicted_code: str, exec_result: ExecutionResult,
                     strong_llm: StrongLLM) -> tuple[str, str]:
    """``(label, rationale)`` for one completion; ``("Pass", "")`` when it passed."""
    group, subgroup, rationale = _classify(task, predicted_code, exec_result,
                                           extract_api_call(predicted_code or "",
                                                            task.test_harness.target_api or ""),
                                           strong_llm)
    if group == "OK":
        return PASS, ""
    return to_label(group, subgroup), rationale


def _classify(
    task: Task,
    predicted_code: str,
    exec_result: ExecutionResult,
    ast_extract: _ASTExtract,
    strong_llm: StrongLLM,
) -> tuple[str, str, str]:
    """``(group, api_subgroup, rationale)``. If the judge fails or returns malformed output,
    the group falls back to W_LOGIC when the target's leaf name appears in the code, else W_API.
    """
    if exec_result.passed:
        return "OK", "", ""

    target_api = task.test_harness.target_api or ""
    target_leaf = target_api.split(".")[-1]
    target_top = target_api.split(".")[0] if target_api else ""
    target_mentioned = bool(target_leaf) and target_leaf in (predicted_code or "")
    undefined_name: str | None = None

    if exec_result.timed_out:
        # Timeouts go to the judge with a "Timeout" exception type.
        exception_type = "Timeout"
        stderr_snippet = (exec_result.stderr or "").strip()[-500:] or "<no stderr (timeout)>"
    else:
        exception_type = extract_exception_type(exec_result.stderr)
        stderr_snippet = exec_result.stderr.strip()[-500:] if exec_result.stderr else ""

        # Deterministic pre-checks: cheap, high-confidence cases skip the LLM.

        # The monitor assertion opens every execution test and fails only when the target
        # API was never invoked at runtime: ground truth about API selection.
        if _SPY_ASSERTION_RE.search(exec_result.stderr or ""):
            rat = (f"target-API spy recorded zero calls: the solution never "
                   f"invoked {target_api or 'the target API'}")
            return "W_API", "W_API_SELECT", rat

        if exception_type in ("SyntaxError", "IndentationError"):
            cls = "W_API"
            rat = f"{exception_type}: model failed to emit valid code"
            return cls, _subclassify_w_api(cls, exception_type, False, rat, target_api, ast_extract.api_called_name), rat

        undefined_name = extract_undefined_name(exec_result.stderr)
        if undefined_name and target_leaf and (undefined_name == target_leaf or undefined_name == target_top):
            cls = "W_API"
            rat = f"NameError on target API symbol '{undefined_name}' — model did not import the call"
            return cls, _subclassify_w_api(cls, exception_type, False, rat, target_api, ast_extract.api_called_name), rat

    prompt = _CLASSIFIER_TEMPLATE.format(
        description=(task.description or "")[:400],
        target_api=target_api or "<none>",
        reference_call=_format_reference_call(task),
        predicted_code=predicted_code[:1200] if predicted_code else "<empty>",
        api_called_leaf=ast_extract.api_called_name or "<none>",
        kwargs=json.dumps(ast_extract.kwargs_extracted, ensure_ascii=False)[:400],
        exception_type=exception_type or "<none>",
        undefined_name=undefined_name or "<none>",
        stderr_snippet=stderr_snippet or "<empty>",
    )

    try:
        raw = strong_llm.generate(
            prompt=prompt,
            system=_CLASSIFIER_SYSTEM,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as e:
        cls, rat = _fallback(target_mentioned, f"classifier error: {type(e).__name__}")
        return cls, _subclassify_w_api(cls, exception_type, target_mentioned, rat, target_api, ast_extract.api_called_name), rat

    cls, rat = _parse_classifier_output(raw, target_mentioned)
    return cls, _subclassify_w_api(cls, exception_type, target_mentioned, rat, target_api, ast_extract.api_called_name), rat


def _fallback(target_mentioned: bool, reason: str) -> tuple[str, str]:
    """Deterministic choice when the LLM judge can't produce a usable label."""
    if target_mentioned:
        return "W_LOGIC", f"{reason}; target API referenced so defaulting to W_LOGIC"
    return "W_API", f"{reason}; target API not referenced so defaulting to W_API"


def _parse_classifier_output(raw: str, target_mentioned: bool) -> tuple[str, str]:
    if not raw:
        return _fallback(target_mentioned, "empty classifier output")
    # Strip code fences if present.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return _fallback(target_mentioned, f"no JSON: {raw[:80]}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _fallback(target_mentioned, f"json parse: {raw[:80]}")
    cls = str(obj.get("class", "")).strip()
    rationale = str(obj.get("rationale", "")).strip()[:240]
    if cls not in JUDGE_GROUPS:
        return _fallback(target_mentioned, f"unusable class '{cls}'")
    return cls, rationale
