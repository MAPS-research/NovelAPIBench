"""Task generation (paper Section 3.2, Appendix B.2 "Stage 3").

One GPT-5-mini call per bundle generates three tasks (easy: a direct call with simple inputs;
medium: the API as one step of a multi-step pipeline; hard: a feature built around it) from
the API name, signature, parameter types, return type, up to three examples (executed ones
first) and M. Signature-modified APIs get the old signature and a change summary and must
exercise the change. Bundles without an executed example get three easy variants instead
(a minimal call, different inputs, an edge case).

A generated task is rejected when its description contains the API's leaf name, when neither
the masked region nor the reference mentions the API, or when the reference solution does not
run; each surviving task receives an execute-then-assert harness (``harness``). If every task
of a response is rejected, the call is repeated once at a lower temperature, and if the
descriptions leaked the name, a rewrite of the descriptions is requested.

The prompts are the ones used to build the released benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from typing import Literal

from omegaconf import DictConfig

from novelapibench.construction.stage3.harness import generate_test_harness
from novelapibench.evaluation.harness import module_import_for_api
from novelapibench.llm.strong import StrongLLM
from novelapibench.log import logger
from novelapibench.runtime.sandbox import execute_code
from novelapibench.schemas import KnowledgeBundle, Task

_SYSTEM = (
    "You are an expert Python software engineer. "
    "Generate realistic, self-contained coding benchmark tasks."
)

_GENERATE_TASKS_PROMPT = """\
Generate 3 Python coding benchmark tasks of increasing difficulty that exercise a specific API. \
The tasks will test whether an LLM can correctly use a newly-released API it has never seen before.

API Information:
- Full name: {api_name}
- Signature: {signature}
- Parameters: {parameters}
- Return type: {return_type}
- Usage examples:
{examples}
{examples_note}
Conceptual background:
{mechanism}

Difficulty levels:
1. easy   — Directly call {api_name} with simple, minimal inputs to produce a straightforward result.
2. medium — Use {api_name} as one step in a multi-step data processing or computation pipeline.
3. hard   — Implement a non-trivial feature or algorithm where {api_name} is a key building block.

Ecological validity (applies to ALL tasks):
- Frame each task around a **realistic software-engineering scenario** a practitioner would \
actually write code for — not a standalone toy exercise. Concrete examples, by library domain:
    * data / scientific: loading a real-looking dataset and preparing it for analysis, \
cleaning messy records from a survey / log / sensor stream, building a summary report
    * ML: training-loop preprocessing, post-hoc evaluation of model outputs, \
augmenting training data, production inference pipeline
    * web / infra: handling a user request, processing a batch job, integrating with an \
external service
- Fixtures must look like plausible real data (named columns that make sense for the domain, \
value ranges that would occur in production, realistic column dtypes). Avoid single-letter \
column names, 1-2 element inputs, or obviously synthetic placeholders unless the API \
genuinely needs them.
- The scenario must make the target API a *natural* choice for the job, not a forced fit. \
If you can't construct a realistic scenario that motivates this specific API, fall back to \
an easy straightforward call rather than a contrived medium/hard case.

Rules that apply to ALL tasks:
- "description" must NOT mention "{api_leaf}" or any recognisable variant of the function/class name. \
  This includes plurals, gerunds, past tense, and hyphenated forms \
  (e.g. if leaf is "remap" avoid "remap", "remaps", "remapping", "remapped", "re-map"; \
  if leaf is "rotate_file" avoid "rotate_file", "rotate file", "rotates file", "rotating file"). \
  Describe the task goal, not its Python implementation.
- "description" should be **as specific as possible so that the target API is the unique natural choice**. \
  Underspecified descriptions ("compute a rank-based association", "preprocess the text", \
  "encode the data", "fit a regression") often have several sibling APIs that would also \
  satisfy them, which causes unfair failures when the solver picks a sibling. \
  Pin down the exact statistic, output structure, algorithm name, distinctive parameter, \
  or behaviour the API produces — even if that requires domain terminology. \
  Example: for `scipy.stats.kendalltau`, write "compute Kendall's τ coefficient and its \
  p-value for the two rankings" — NOT "compute the rank-based association between the two \
  ratings" (which `spearmanr` also satisfies). Named statistics, Greek letters, algorithm \
  names, and specific output-type names are encouraged and do NOT trigger the leaf-name \
  guard above (which only blocks the literal `{api_leaf}` identifier as a substring).
- "reference_solution" must be fully self-contained: include every import and define every variable it uses.
- "reference_solution" must be sequential code that directly uses `{api_name}` inline — do NOT wrap it in a named helper function or class method. Write the actual operations at the top level.
- "context_code" is the setup code (imports, variable definitions) that is visible to the solver before \
the masked region; it should set up whatever the solver needs.
- "masked_region" is the specific lines the solver must predict — typically the API call and any \
immediately surrounding logic that depends on it.
- CRITICAL: You MUST use the EXACT fully-qualified name `{api_name}` in masked_region and \
reference_solution. Do NOT substitute with any alias, submodule, or shortened form \
(e.g. do not replace `{api_name}` with a different namespace).

Respond with a JSON array of exactly 3 objects and nothing else:
[
  {{
    "difficulty": "easy",
    "description": "...",
    "context_code": "...",
    "masked_region": "...",
    "reference_solution": "..."
  }},
  {{
    "difficulty": "medium",
    "description": "...",
    "context_code": "...",
    "masked_region": "...",
    "reference_solution": "..."
  }},
  {{
    "difficulty": "hard",
    "description": "...",
    "context_code": "...",
    "masked_region": "...",
    "reference_solution": "..."
  }}
]
"""

_GENERATE_TASKS_MODIFIED_PROMPT = """\
Generate 3 Python coding benchmark tasks of increasing difficulty that exercise \
the **changed behaviour** of a modified API. The API already existed in the previous \
library version but its signature or defaults have changed. Tasks must specifically \
target the modifications so that code written with the OLD signature would fail or \
produce incorrect results.

API Information:
- Full name: {api_name}
- NEW signature: {signature}
- NEW parameters: {parameters}
- Return type: {return_type}
- OLD signature (previous version): {old_signature}
- OLD parameters: {old_parameters}
- Usage examples (new version):
{examples}
{examples_note}
Conceptual background:
{mechanism}

What changed (you MUST build tasks around these differences):
{change_summary}

Difficulty levels:
1. easy   — Directly call {api_name} using a NEW parameter or new default value that \
did not exist in the old version. The task must fail if the old signature is used.
2. medium — Use {api_name} in a multi-step pipeline where the changed behaviour \
(new param, changed default, removed param) is critical to producing the correct result.
3. hard   — Implement a non-trivial feature that relies on the new API behaviour. \
Code using the old signature should produce a detectably wrong result or raise an error.

Ecological validity (applies to ALL tasks):
- Frame each task around a **realistic software-engineering scenario** a practitioner would \
actually write code for — not a standalone toy exercise (data/science: real-looking datasets \
and analysis workflows; ML: training-loop preprocessing, evaluation, production inference; \
web/infra: request handling, batch jobs, external integrations).
- Fixtures must look like plausible real data (named columns, realistic value ranges and \
dtypes). Avoid single-letter columns, 1-2 element inputs, or obviously synthetic placeholders \
unless the API genuinely needs them.
- The scenario must make the target API a *natural* choice for the job, not a forced fit.

Rules that apply to ALL tasks:
- "description" must NOT mention "{api_leaf}" or any recognisable variant of the function/class name. \
  This includes plurals, gerunds, past tense, and hyphenated forms. \
  Describe the task goal, not its Python implementation.
- "description" should be **as specific as possible so that the target API is the unique natural choice**. \
  Underspecified descriptions often have several sibling APIs that would also satisfy them, \
  which causes unfair failures when the solver picks a sibling. Pin down the exact \
  statistic, output structure, algorithm name, distinctive parameter, or behaviour the API \
  produces — even if that requires domain terminology. \
  Example: for `scipy.stats.kendalltau`, write "compute Kendall's τ coefficient and its \
  p-value for the two rankings" — NOT "compute the rank-based association between the two \
  ratings" (which `spearmanr` also satisfies). Named statistics, Greek letters, algorithm \
  names, and specific output-type names are encouraged and do NOT trigger the leaf-name \
  guard above (which only blocks the literal `{api_leaf}` identifier as a substring).
- "reference_solution" must be fully self-contained: include every import and define every variable it uses.
- "reference_solution" must be sequential code that directly uses `{api_name}` inline — do NOT wrap it in a named helper function or class method.
- "context_code" is the setup code visible to the solver before the masked region.
- "masked_region" is the specific lines the solver must predict.
- CRITICAL: You MUST use the EXACT fully-qualified name `{api_name}` in masked_region and \
reference_solution.
- CRITICAL: Every task MUST exercise at least one changed aspect of the API (new parameter, \
changed default, etc.). A task that could be solved with the old signature is invalid.

Respond with a JSON array of exactly 3 objects and nothing else:
[
  {{
    "difficulty": "easy",
    "description": "...",
    "context_code": "...",
    "masked_region": "...",
    "reference_solution": "..."
  }},
  {{
    "difficulty": "medium",
    "description": "...",
    "context_code": "...",
    "masked_region": "...",
    "reference_solution": "..."
  }},
  {{
    "difficulty": "hard",
    "description": "...",
    "context_code": "...",
    "masked_region": "...",
    "reference_solution": "..."
  }}
]
"""

# Bundles whose examples all failed to execute standalone: three easy variants.
_GENERATE_EASY_ONLY_PROMPT = """\
Generate 3 Python coding benchmark tasks that each directly exercise a specific API. \
All 3 tasks must be of "easy" difficulty — a single direct call to the API with minimal, \
concrete inputs producing a straightforward result.

API Information:
- Full name: {api_name}
- Signature: {signature}
- Parameters: {parameters}
- Return type: {return_type}
- Usage examples:
{examples}
{examples_note}
Conceptual background:
{mechanism}

This API's only known usage examples could not be executed in isolation during \
validation (they depend on external weights, network, or a specific runtime \
environment).  Therefore do NOT try to write multi-step pipelines or hard \
algorithmic tasks around it.  Instead, produce 3 minimally-differing easy tasks \
that each call {api_name} in the simplest way that can run standalone:

1. easy_1 — the canonical minimal instantiation / call with small, cheap inputs.
2. easy_2 — same shape as easy_1 but with different concrete input values.
3. easy_3 — a boundary / edge-case variant (e.g. empty or minimal-size input).

Rules that apply to ALL tasks:
- "description" must NOT mention "{api_leaf}" or any recognisable variant of the function/class name.
- "description" should be **as specific as possible so that the target API is the unique natural choice**. \
  Underspecified descriptions often have several sibling APIs that would also satisfy them, \
  which causes unfair failures when the solver picks a sibling. Pin down the exact \
  statistic, output structure, algorithm name, distinctive parameter, or behaviour the API \
  produces — even if that requires domain terminology. \
  Example: for `scipy.stats.kendalltau`, write "compute Kendall's τ coefficient and its \
  p-value for the two rankings" — NOT "compute the rank-based association between the two \
  ratings" (which `spearmanr` also satisfies). Named statistics, Greek letters, algorithm \
  names, and specific output-type names are encouraged and do NOT trigger the leaf-name \
  guard above (which only blocks the literal `{api_leaf}` identifier as a substring).
- "reference_solution" must be fully self-contained: include every import and define every variable it uses.
- "reference_solution" must be sequential code that directly uses `{api_name}` inline — do NOT wrap it in a named helper function or class method.
- Prefer tiny synthetic inputs (small tensors, short strings, in-memory data) over anything that requires a network, GPU, or large model weights.
- "context_code" is the setup code visible to the solver before the masked region.
- "masked_region" is the specific lines the solver must predict.
- CRITICAL: You MUST use the EXACT fully-qualified name `{api_name}` in masked_region and reference_solution.
- Every task MUST have "difficulty" set to "easy".

Respond with a JSON array of exactly 3 objects and nothing else:
[
  {{"difficulty": "easy", "description": "...", "context_code": "...", "masked_region": "...", "reference_solution": "..."}},
  {{"difficulty": "easy", "description": "...", "context_code": "...", "masked_region": "...", "reference_solution": "..."}},
  {{"difficulty": "easy", "description": "...", "context_code": "...", "masked_region": "...", "reference_solution": "..."}}
]
"""

_DESC_VIOLATION_RETRY_PROMPT = """\
The previous task descriptions were rejected because they contained the forbidden word \
"{api_leaf}" (or a grammatical variant). Rewrite ALL 3 tasks with new descriptions that \
describe the task goal WITHOUT using "{api_leaf}", "{api_leaf_variants}". \
Keep reference_solution, context_code, and masked_region identical to before.

Previous response (to reuse except for descriptions):
{prev_response}

Respond with a JSON array of exactly 3 objects:
[
  {{"difficulty": "easy", "description": "...", "context_code": "...", "masked_region": "...", "reference_solution": "..."}},
  {{"difficulty": "medium", "description": "...", "context_code": "...", "masked_region": "...", "reference_solution": "..."}},
  {{"difficulty": "hard", "description": "...", "context_code": "...", "masked_region": "...", "reference_solution": "..."}}
]
"""

_DIFFICULTIES: tuple[Literal["easy", "medium", "hard"], ...] = ("easy", "medium", "hard")

_NOTE_STATIC_ONLY = (
    "\nNote: the examples above did NOT execute cleanly during "
    "validation — they are structurally valid but rely on external "
    "weights, network access, or specific runtime setup.  Do NOT "
    "copy them verbatim.  Write reference solutions that run "
    "standalone with small synthetic inputs.\n"
)
_NOTE_NO_EXAMPLES = (
    "\nNote: no working usage examples are available.  Infer the "
    "minimal self-contained call from the signature and parameters.\n"
)


def summarise_changes(old_sig: str, old_params: list[dict], new_sig: str,
                      new_params: list[dict]) -> str:
    """Added / removed parameters and changed defaults between two signatures."""
    old_names = {p["name"] for p in old_params if p.get("name") not in ("self", "cls")}
    new_names = {p["name"] for p in new_params if p.get("name") not in ("self", "cls")}
    old_defaults = {p["name"]: p.get("default_value") or p.get("default") for p in old_params}
    new_defaults = {p["name"]: p.get("default_value") or p.get("default") for p in new_params}
    lines = []
    if added := new_names - old_names:
        lines.append(f"- New parameters added: {', '.join(sorted(added))}")
    if removed := old_names - new_names:
        lines.append(f"- Parameters removed: {', '.join(sorted(removed))}")
    for name in sorted(old_names & new_names):
        od, nd = old_defaults.get(name), new_defaults.get(name)
        if od != nd:
            lines.append(f"- Parameter '{name}' default changed: {od!r} → {nd!r}")
    if not lines:
        lines = [f"- Old signature: {old_sig}", f"- New signature: {new_sig}",
                 "- (Exact change not determined — tasks should test new parameter behaviour)"]
    return "\n".join(lines)


def leaf_variants(leaf: str) -> list[str]:
    """Grammatical variants of the API's leaf name listed as forbidden in the rewrite prompt."""
    base = leaf.replace("_", " ").lower()
    variants = {leaf, base, leaf.replace("_", "").lower()}
    for w in (leaf, base):
        variants.update([w + "s", w + "ed", w + "ing", w + "er"])
    return sorted(variants)


def build_prompt(bundle: KnowledgeBundle, easy_only: bool) -> str:
    params = ", ".join(f"{p.name}: {p.type or 'any'}" for p in bundle.s_param[:8]) or "N/A"
    executed = [ex for ex in bundle.examples if ex.status == "executed"]
    static = [ex for ex in bundle.examples if ex.status != "executed"]
    ordered = (executed + static)[:3]
    note = "" if executed else _NOTE_STATIC_ONLY if static else _NOTE_NO_EXAMPLES
    fields = dict(api_name=bundle.api_name, api_leaf=bundle.api_name.split(".")[-1],
                  signature=f"{bundle.api_name}{bundle.signature}", parameters=params,
                  return_type=bundle.return_type or "unknown",
                  examples="\n".join(ex.code for ex in ordered) if ordered else "N/A",
                  examples_note=note, mechanism=bundle.mechanism.text or "N/A")
    if easy_only:
        return _GENERATE_EASY_ONLY_PROMPT.format(**fields)
    if bundle.is_modified and bundle.old_signature:
        old_params = "N/A"
        if bundle.old_parameters:
            old_params = ", ".join(f"{p['name']}: {p.get('annotation') or 'any'}"
                                   for p in bundle.old_parameters[:8]
                                   if p.get("name") not in ("self", "cls")) or "N/A"
        change = summarise_changes(bundle.old_signature, bundle.old_parameters or [],
                                   f"{bundle.api_name}{bundle.signature}",
                                   [p.model_dump() for p in bundle.s_param])
        return _GENERATE_TASKS_MODIFIED_PROMPT.format(
            **fields, old_signature=f"{bundle.api_name}{bundle.old_signature}",
            old_parameters=old_params, change_summary=change)
    return _GENERATE_TASKS_PROMPT.format(**fields)


def _json_array(raw: str) -> list | None:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return items if isinstance(items, list) else None


def reference_runs(bundle: KnowledgeBundle, context_code: str, reference: str, cfg: DictConfig,
                   env_python: str) -> tuple[bool, str | None]:
    """Does ``import <module>`` + context + reference run in the library's environment?"""
    s3 = stage3_settings(bundle.library, cfg)
    setup = "\n\n".join(filter(None, [module_import_for_api(bundle.api_name), context_code.strip()]))
    r = execute_code("\n\n".join(filter(None, [setup, reference.strip()])),
                     timeout=s3["exec_timeout"], max_memory_mb=s3["memory_mb"],
                     env_python=env_python, pid_namespace=bool(cfg.construction.sandbox.pid_namespace))
    if r.passed:
        return True, None
    return False, (r.stderr[-500:] if r.stderr else r.error_msg) or "execution_failed"


def stage3_settings(library: str, cfg: DictConfig) -> dict:
    """Stage-3 sandbox limits, with the library's overrides."""
    from novelapibench.config import load_library_config
    lc = load_library_config(library).get("construction", {})
    s3 = cfg.construction.stage3
    return {"exec_timeout": int(lc.get("stage3_exec_timeout_seconds", s3.exec_timeout_seconds)),
            "validation_timeout": int(lc.get("stage3_validation_timeout_seconds",
                                             s3.validation_timeout_seconds)),
            "memory_mb": int(lc.get("stage3_sandbox_memory_mb", s3.sandbox_memory_mb))}


def parse_tasks(raw: str, bundle: KnowledgeBundle, cfg: DictConfig, llm: StrongLLM,
                env_python: str) -> list[Task]:
    items = _json_array(raw)
    if not items:
        return []
    leaf = bundle.api_name.split(".")[-1]
    tasks = []
    for item in items:
        if not isinstance(item, dict) or item.get("difficulty", "") not in _DIFFICULTIES:
            continue
        difficulty = item["difficulty"]
        description = item.get("description", "").strip()
        reference = item.get("reference_solution", "").strip()
        if not description or not reference or leaf.lower() in description.lower():
            continue
        context = textwrap.dedent(item.get("context_code", "")).strip()
        masked = textwrap.dedent(item.get("masked_region", "")).strip()
        reference = textwrap.dedent(reference).strip()
        if bundle.api_name not in masked + reference and leaf not in masked + reference:
            continue
        ok, err = reference_runs(bundle, context, reference, cfg, env_python)
        if not ok:
            logger.debug(f"{bundle.api_name} [{difficulty}]: reference fails: {(err or '')[:200]}")
            continue
        try:
            harness = generate_test_harness(bundle.api_name, reference, description, llm, cfg,
                                            context, env_python, bundle.return_type, bundle.library)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{bundle.api_name} [{difficulty}]: harness generation failed: {exc}")
            continue
        digest = hashlib.md5(f"{bundle.api_name}:{difficulty}:{description[:40]}".encode()).hexdigest()
        tasks.append(Task(task_id=f"llm_{digest[:12]}", api_name=bundle.api_name,
                          library=bundle.library, domain=bundle.domain, difficulty=difficulty,
                          description=description, context_code=context, masked_region=masked,
                          reference_solution=reference, test_harness=harness))
    return tasks


def generate_tasks(bundle: KnowledgeBundle, cfg: DictConfig, llm: StrongLLM,
                   env_python: str) -> list[Task]:
    """Up to three tasks (with harnesses) for one bundle."""
    s3 = cfg.construction.stage3
    easy_only = not any(ex.status == "executed" for ex in bundle.examples)
    prompt = build_prompt(bundle, easy_only)
    leaf = bundle.api_name.split(".")[-1]
    last_raw = ""
    for temperature in (float(s3.temperature), float(s3.retry_temperature)):
        try:
            last_raw = llm.generate(prompt, system=_SYSTEM, temperature=temperature)
            tasks = parse_tasks(last_raw, bundle, cfg, llm, env_python)
            if tasks:
                return tasks
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{bundle.api_name}: generation attempt failed: {exc}")
    items = _json_array(last_raw) if last_raw else None
    if items and any(isinstance(i, dict) and leaf.lower() in i.get("description", "").lower()
                     for i in items):
        try:
            retry = _DESC_VIOLATION_RETRY_PROMPT.format(
                api_leaf=leaf, api_leaf_variants=", ".join(leaf_variants(leaf)),
                prev_response=last_raw[:3000])
            raw = llm.generate(retry, system=_SYSTEM, temperature=float(s3.rewrite_temperature))
            return parse_tasks(raw, bundle, cfg, llm, env_python)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"{bundle.api_name}: description rewrite failed: {exc}")
    return []
