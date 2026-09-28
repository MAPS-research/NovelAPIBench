"""Execute-then-assert harness construction (Appendix B.3, "Harness construction").

1. GPT-5-mini writes six self-contained scenarios that call the reference solution's API on
   concrete inputs (two basic, two with changed inputs, two edge cases), each ending with a
   print protocol that describes ``result`` (type, shape, dtype, length, value, non-zero, sum).
2. The scenarios run after ``import <module>`` + context + reference solution (a preamble that
   fails on an import error rejects every scenario; one that fails otherwise is dropped so
   the self-contained scenarios run alone). Failed scenarios are re-prompted once with their
   errors when fewer than four succeeded. At least two must succeed.
3. Assertions are built from the recorded descriptions: layer 1 checks a basic scenario's
   shape / length / value and non-emptiness, layer 2 checks that changed inputs change the
   output, layer 3 checks an edge scenario.
4. A layer that still passes when the target is replaced by a stub returning zeros or
   ``None`` is discarded.
5. The target-call monitor (``evaluation.harness``) becomes ``setup_code``, and each kept layer
   is prefixed with the monitor assertion. The task's ``execution_test`` is layer 1.

The prompts are the ones used to build the released benchmark.
"""

from __future__ import annotations

import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig

from novelapibench.evaluation.harness import (
    build_monitor_check, build_monitor_setup, module_import_for_api)
from novelapibench.llm.strong import StrongLLM
from novelapibench.log import logger
from novelapibench.runtime.sandbox import execute_code
from novelapibench.schemas import TestHarness

# Print protocol appended to every scenario: describes `result` as one JSON line.
_RESULT_DESCRIBE_CODE = """\
import json as _json_protocol

def _describe_result(_obj):
    _desc = {"type": type(_obj).__name__}
    if hasattr(_obj, "shape"):
        try:
            _desc["shape"] = repr(_obj.shape)
        except Exception:
            pass
    if hasattr(_obj, "dtype"):
        try:
            _desc["dtype"] = str(_obj.dtype)
        except Exception:
            pass
    if hasattr(_obj, "__len__"):
        try:
            _desc["len"] = len(_obj)
        except Exception:
            pass
    if isinstance(_obj, (int, float, complex, bool)):
        _desc["value"] = repr(_obj)
    if hasattr(_obj, "any") and callable(getattr(_obj, "any")):
        try:
            _desc["nonzero"] = bool(_obj.any())
        except Exception:
            pass
    elif hasattr(_obj, "__len__"):
        try:
            _desc["nonzero"] = len(_obj) > 0
        except Exception:
            pass
    if hasattr(_obj, "sum") and callable(getattr(_obj, "sum")):
        try:
            _desc["sum"] = float(_obj.sum())
        except Exception:
            pass
    return _desc

print("RESULT_DESC:", _json_protocol.dumps(_describe_result(result)))
"""

_SCENARIO_PROMPT = """\
You are writing test scenarios for a Python coding benchmark.

The target API being tested: `{api_name}`
Return type of the target API: {return_type}

Task description (what the model is asked to implement):
{description}

Reference solution (complete, correct implementation):
```python
{reference_code}
```

Generate exactly 6 self-contained Python test scenarios for the above code.
Each scenario must:
1. Import any needed modules at the top.
2. Call the target API / function / class from the reference solution with CONCRETE inputs.
3. Assign the result to a variable named exactly `result`.
4. End with these EXACT lines (copy verbatim — do NOT modify):

```python
{describe_code}
```

Generate exactly these 6 scenario names:
- "basic_1": standard call with typical inputs
- "basic_2": standard call with different (but equally typical) inputs
- "changed_input_1": same structure as basic_1 but different input values
- "changed_input_2": same structure as basic_2 but different input values
- "edge_1": an edge case (e.g. empty input, minimal size, boundary value)
- "edge_2": another edge case (e.g. large input, maximal value, unusual-but-valid input)

CRITICAL RULES:
- Each scenario is fully self-contained (no shared state between scenarios).
- Do NOT include assert statements — only the print protocol lines.
- Do NOT redefine the target API/class — it is already defined by the reference solution.
- The variable `result` must hold the direct return value of the call.
- If the API returns a tuple/list, keep `result` as-is (the print protocol handles it).

Return ONLY a JSON object (no markdown fences, no explanation):
{{
  "basic_1": "<full self-contained python code string>",
  "basic_2": "<full self-contained python code string>",
  "changed_input_1": "<full self-contained python code string>",
  "changed_input_2": "<full self-contained python code string>",
  "edge_1": "<full self-contained python code string>",
  "edge_2": "<full self-contained python code string>"
}}
"""

_RETRY_PROMPT = """\
Your previous test scenarios for `{api_name}` failed with errors:

{failures}

Re-read the reference solution carefully:
```python
{reference_code}
```

Generate fixed versions of ONLY the failed scenarios.
Each scenario must end with these EXACT lines:
```python
{describe_code}
```

Return ONLY a JSON object with the same scenario names as keys:
{{"scenario_name": "fixed python code", ...}}
"""

_RESULT_MARKER = "RESULT_DESC:"


def _parse_scenario_stdout(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER):].strip())
            except json.JSONDecodeError:
                return None
    return None


def _strip_print_protocol(code: str) -> str:
    """Remove the describe helper (import, function, print line) from a scenario."""
    lines, out, i = code.splitlines(), [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "import json as _json_protocol":
            i += 1
            continue
        if s.startswith("def _describe_result("):
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][0] in (" ", "\t")):
                i += 1
            continue
        if (s.startswith('print("RESULT_DESC:') or s.startswith("print('RESULT_DESC:")
                or ("_describe_result" in s and s.startswith("print("))):
            i += 1
            continue
        out.append(lines[i])
        i += 1
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _json_object(raw: str) -> dict:
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.startswith("```")).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    data = json.loads(raw)
    return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}


def generate_scenarios(api_name: str, description: str, reference: str, llm: StrongLLM,
                       temperature: float, return_type: str | None = None) -> dict[str, str]:
    prompt = _SCENARIO_PROMPT.format(api_name=api_name, return_type=return_type or "unknown",
                                     description=description, reference_code=reference[:4000],
                                     describe_code=_RESULT_DESCRIBE_CODE.strip())
    try:
        return _json_object(llm.generate(prompt, temperature=temperature))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"scenario generation failed: {exc}")
        return {}


def retry_scenarios(api_name: str, reference: str, failed: dict[str, tuple[str, str]],
                    llm: StrongLLM, temperature: float) -> dict[str, str]:
    blocks = []
    for name, (code, err) in failed.items():
        tail = err.strip().splitlines()
        short = "\n".join(tail[-6:]) if len(tail) > 6 else err.strip()
        blocks.append(f'Scenario "{name}":\n  Code:\n{code[:400]}\n  Error:\n{short}')
    prompt = _RETRY_PROMPT.format(api_name=api_name, failures="\n\n".join(blocks),
                                  reference_code=reference[:3000],
                                  describe_code=_RESULT_DESCRIBE_CODE.strip())
    try:
        return _json_object(llm.generate(prompt, temperature=temperature))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"scenario retry failed: {exc}")
        return {}


def execute_scenarios(api_name: str, reference: str, context_code: str,
                      scenarios: dict[str, str], limits: dict, use_mock: bool, workers: int,
                      env_python: str, pid_namespace: bool = False
                      ) -> tuple[dict[str, dict], dict[str, tuple[str, str]]]:
    """``(succeeded {name: {code, captured}}, failed {name: (code, error)})``."""
    timeout, memory = limits["exec_timeout"], limits["memory_mb"]
    preamble = "\n\n".join(filter(None, [module_import_for_api(api_name), context_code.strip(),
                                         reference.strip()]))
    helper = _RESULT_DESCRIBE_CODE.replace(
        'print("RESULT_DESC:", _json_protocol.dumps(_describe_result(result)))', "").strip()
    check = execute_code(preamble, timeout=timeout, max_memory_mb=memory, env_python=env_python,
                         use_mock_imports=use_mock, pid_namespace=pid_namespace)
    if not check.passed:
        err = str(check.stderr or check.error_msg or "preamble_failed")
        if any(k in err for k in ("ImportError", "ModuleNotFoundError", "No module named")):
            return {}, {n: (c, f"preamble_failed: {err[-200:]}") for n, c in scenarios.items()}
        preamble = ""   # a runtime error in the preamble: run the scenarios on their own

    def run(name: str, code: str):
        r = execute_code("\n\n".join(filter(None, [preamble, helper, code])), timeout=timeout,
                         max_memory_mb=memory, env_python=env_python, use_mock_imports=use_mock,
                         pid_namespace=pid_namespace)
        if not r.passed:
            return name, False, (code, str(r.stderr or r.error_msg or "execution_failed"))
        captured = _parse_scenario_stdout(r.stdout)
        if captured is None:
            return name, False, (code, "no RESULT_DESC in stdout")
        return name, True, {"code": code, "captured": captured}

    succeeded, failed = {}, {}
    with ThreadPoolExecutor(max_workers=min(workers, len(scenarios))) as pool:
        for fut in as_completed([pool.submit(run, n, c) for n, c in scenarios.items()]):
            name, ok, payload = fut.result()
            (succeeded if ok else failed)[name] = payload
    return succeeded, failed


def _structural_test(scenario_code: str, captured: dict) -> str:
    code = _strip_print_protocol(scenario_code)
    lines = [code]
    if captured.get("shape"):
        shape = captured["shape"]
        lines.append(f"assert hasattr(result, 'shape') and repr(result.shape) == {shape!r}, "
                     f"f\"shape mismatch: {{repr(result.shape)}} != {shape!r}\"")
    elif "len" in captured:
        n = captured["len"]
        lines.append(f"assert hasattr(result, '__len__') and len(result) == {n}, "
                     f"f\"length mismatch: {{len(result)}} != {n}\"")
    elif "value" in captured:
        v = captured["value"]
        lines.append(f"assert repr(result) == {v!r}, f\"value mismatch: {{repr(result)}} != {v!r}\"")
    if captured.get("nonzero") is True:
        if captured.get("type", "") in ("Tensor",) or captured.get("shape"):
            lines.append("assert result is not None and (result.any() if hasattr(result, 'any') "
                         "else len(result) > 0), \"result is empty/all-zeros\"")
        else:
            lines.append("assert result is not None and bool(result), \"result is falsy\"")
    return "\n".join(lines)


def _comparison_test(basic_code: str, changed_code: str) -> str:
    return "\n".join([
        "# --- Run 1: basic input ---",
        _strip_print_protocol(basic_code),
        "_result1 = result",
        "",
        "# --- Run 2: changed input ---",
        _strip_print_protocol(changed_code),
        "_result2 = result",
        "",
        "# Assert outputs differ with different inputs",
        "try:",
        "    import numpy as _np",
        "    if hasattr(_result1, 'shape') and _result1.shape == _result2.shape:",
        "        assert not _np.allclose(_result1, _result2) if hasattr(_result1, 'numpy') else (_result1 != _result2).any(), \\",
        "            \"output does not change with different inputs\"",
        "    else:",
        "        assert _result1 is not _result2, \"same result object returned\"",
        "except ImportError:",
        "    assert _result1 is not _result2 or str(_result1) != str(_result2), \\",
        "        \"output does not change with different inputs\"",
    ])


def build_test_layers(succeeded: dict[str, dict]) -> dict[str, str]:
    layers = {"test_layer1": "", "test_layer2": "", "test_layer3": ""}
    basic = [v for k, v in succeeded.items() if k.startswith("basic_")]
    changed = [v for k, v in succeeded.items() if k.startswith("changed_input_")]
    edge = [v for k, v in succeeded.items() if k.startswith("edge_")]
    everything = list(succeeded.values())
    src1 = basic[0] if basic else (everything[0] if everything else None)
    if src1:
        layers["test_layer1"] = _structural_test(src1["code"], src1["captured"])
    if basic and changed:
        layers["test_layer2"] = _comparison_test(basic[0]["code"], changed[0]["code"])
    elif len(basic) >= 2:
        layers["test_layer2"] = _comparison_test(basic[0]["code"], basic[1]["code"])
    elif len(everything) >= 2:
        layers["test_layer2"] = _comparison_test(everything[0]["code"], everything[1]["code"])
    src3 = edge[0] if edge else (basic[1] if len(basic) >= 2 else None)
    if src3 is None and len(everything) >= 2:
        src3 = everything[1]
    if src3:
        layers["test_layer3"] = _structural_test(src3["code"], src3["captured"])
    return layers


def null_stub(reference: str, api_name: str) -> str:
    """The reference followed by a definition that shadows the target with a zero/None stub."""
    leaf = api_name.split(".")[-1]
    is_class = False
    try:
        is_class = any(isinstance(n, ast.ClassDef) and n.name == leaf
                       for n in ast.walk(ast.parse(reference)))
    except SyntaxError:
        pass
    zeros = """\
    for a in args:
        if hasattr(a, 'shape'):
            try:
                import numpy as _np_stub
                return _np_stub.zeros_like(a)
            except ImportError:
                pass
            try:
                import torch as _torch_stub
                return _torch_stub.zeros_like(a)
            except ImportError:
                pass
    return None
"""
    if is_class:
        body = "\n".join("    " + line if line else line for line in zeros.splitlines()) + "\n"
        override = (f"# --- NULL STUB (shadows real {leaf}) ---\nclass {leaf}:\n"
                    "    def __init__(self, *args, **kwargs):\n        pass\n"
                    "    def __call__(self, *args, **kwargs):\n"
                    "        return self.forward(*args, **kwargs) if hasattr(self, 'forward') else None\n"
                    "    def forward(self, *args, **kwargs):\n" + body)
    else:
        override = (f"# --- NULL STUB (shadows real {leaf}) ---\ndef {leaf}(*args, **kwargs):\n"
                    + zeros)
    return reference + "\n\n" + override


def discard_uninformative(layers: dict[str, str], stub_code: str, context_code: str,
                          limits: dict, use_mock: bool, env_python: str,
                          pid_namespace: bool = False) -> dict[str, str]:
    """Layers that fail against the null stub (the others cannot tell a stub from the API)."""
    active = {k: v for k, v in layers.items() if v.strip()}
    if not active:
        return {}

    def check(layer: str, test: str) -> tuple[str, bool]:
        r = execute_code("\n\n".join(filter(None, [context_code.strip(), stub_code, test])),
                         timeout=limits["exec_timeout"], max_memory_mb=limits["memory_mb"],
                         env_python=env_python, use_mock_imports=use_mock, pid_namespace=pid_namespace)
        return layer, r.passed

    informative = {}
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        for fut in as_completed([pool.submit(check, k, v) for k, v in active.items()]):
            layer, passes_stub = fut.result()
            if not passes_stub:
                informative[layer] = active[layer]
    return informative


def generate_test_harness(api_name: str, reference: str, description: str, llm: StrongLLM,
                          cfg: DictConfig, context_code: str, env_python: str,
                          return_type: str | None, library: str) -> TestHarness:
    """The task's harness; empty ``execution_test`` when construction fails (task rejected)."""
    from novelapibench.construction.stage3.tasks import stage3_settings

    hc = cfg.construction.stage3.harness
    limits = stage3_settings(library, cfg)
    pid_ns = bool(cfg.construction.sandbox.pid_namespace)
    use_mock, workers = bool(hc.use_mock_imports), int(hc.scenario_workers)
    scenarios = generate_scenarios(api_name, description, reference, llm,
                                   float(hc.scenario_temperature), return_type)
    total = len(scenarios)
    if not scenarios:
        return TestHarness(target_api=api_name, generation_method="llm_only")
    succeeded, failed = execute_scenarios(api_name, reference, context_code, scenarios, limits,
                                          use_mock, workers, env_python, pid_ns)
    preamble_broken = failed and all("preamble_failed" in err for _, err in failed.values())
    if failed and len(succeeded) < int(hc.retry_below) and not preamble_broken:
        fixed = retry_scenarios(api_name, reference, failed, llm, float(hc.retry_temperature))
        if fixed:
            again, _ = execute_scenarios(api_name, reference, context_code, fixed, limits,
                                         use_mock, workers, env_python, pid_ns)
            succeeded.update(again)
    if len(succeeded) < int(hc.min_succeeded_scenarios):
        return TestHarness(target_api=api_name, generation_method="execute_then_assert_failed",
                           scenarios_total=total, scenarios_succeeded=len(succeeded))
    layers = build_test_layers(succeeded)
    if hc.null_stub_check:
        layers = discard_uninformative(layers, null_stub(reference, api_name), context_code,
                                       limits, use_mock, env_python, pid_ns)
    else:
        layers = {k: v for k, v in layers.items() if v.strip()}
    check = build_monitor_check(api_name)

    def with_check(layer: str) -> str:
        return check + "\n" + layer if layer.strip() else ""

    return TestHarness(setup_code=build_monitor_setup(api_name),
                       execution_test=with_check(layers.get("test_layer1", "")),
                       shape_type_test=with_check(layers.get("test_layer2", "")),
                       mock_test=with_check(layers.get("test_layer3", "")),
                       target_api=api_name, generation_method="execute_then_assert",
                       scenarios_total=total, scenarios_succeeded=len(succeeded))
