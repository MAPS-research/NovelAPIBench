"""Surface information S_name, S_param and usage examples E (Appendix B.2, "Stage 2").

S: GPT-5-mini with web search reads the official documentation for the API and returns its
parameters with descriptions, the return type, and whether the API is primarily file I/O.
The introspected signature is authoritative: parameter names, defaults and annotations come
from it, and the extraction model contributes only descriptions (and types where the source
has no annotation). S_name is the fully qualified name itself.

E: a separate call generates up to three examples from the signature, the parameter
descriptions and the docstring; each is validated in the sandbox (``examples``). If all are
rejected, generation is retried up to twice with the rejection reasons as feedback.

The prompts are the ones used to build the released benchmark.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from omegaconf import DictConfig

from novelapibench.construction.schemas import APIEntry
from novelapibench.construction.stage2.examples import validate_example
from novelapibench.llm.strong import StrongLLM
from novelapibench.log import logger
from novelapibench.schemas import Example, Parameter

_SYSTEM = (
    "You are a precise technical documentation extractor. "
    "Return only valid JSON — no markdown fences, no explanation."
)

_SURFACE_PROMPT = """\
Look up the official Python documentation for the following API and extract its \
K1 (surface) knowledge.

Library : {library}
Version : {version}
API name: {api_name}

Docstring (from source code — use as reference but prefer the official docs):
\"\"\"
{docstring}
\"\"\"

Return a single JSON object with exactly these keys:
{{
  "signature": "( param: Type = default, ... )",
  "parameters": [
    {{"name": "p", "type": "str or null", "default": "value or null", "description": "..."}}
  ],
  "return_type": "str or null",
  "requires_file_io": false
}}

Rules:
- "signature" must be the constructor / function parameter list, excluding self.
- "parameters" must exclude self, *args, **kwargs unless they are meaningful.
- "return_type" is the Python type returned by the function / method (e.g. \
"torch.Tensor", "tuple[Tensor, Tensor]", "None"). For a class, give the type \
of an instance (usually the class's own qualified name). Use null only when \
truly unknown from the docs.
- "requires_file_io" must be true if the API's primary purpose is reading from \
or writing to files, paths, URLs, or streams (e.g. read_csv, to_parquet, \
load_model). Set false for APIs that merely accept optional path arguments but \
whose primary purpose is computation (e.g. to_datetime, fit).
- If a field is unknown, use null / empty list.
- Return ONLY the JSON object."""

_EXAMPLES_PROMPT = """\
Generate up to {n} minimal, self-contained Python code examples that demonstrate \
how to use the API below. These examples are going to be used as training data, \
so every example must be correct.

Library : {library}
Version : {version}
API name: {api_name}
Signature: {api_name}{signature}

Parameters:
{param_lines}

Docstring (authoritative — follow the initialization / usage patterns shown in \
any "Examples" or "Usage" section verbatim if present, instead of inventing a \
simpler pattern):
\"\"\"
{docstring}
\"\"\"

Before writing any example, search the web — specifically the library's \
official GitHub repository — for real-world usage of `{api_name}`. Useful \
places to look (same for every library):
  - The repo's `examples/`, `scripts/`, or `notebooks/` directory.
  - The test suite (`tests/`): unit tests typically construct the object with \
the minimal set of arguments that actually works.
  - The `README.md` and any quickstart or docs pages.
  - Issues or pull requests discussing this symbol if the above yield nothing.
Mirror the idioms you find there (constructor style, required factory, \
expected tensor shapes / dtypes, how the return value is consumed). Do NOT \
invent a plausible-looking pattern when real examples exist.

Requirements for every example:
1. Complete and runnable: all imports included; no placeholder comments such \
as "# your code here".
2. Must actually instantiate or call `{api_name}` exactly as named — do not \
alias it away or use a different API in its place.
3. Never pass arguments that are not listed in the signature above.
4. Input values must respect the declared parameter annotations / types.
5. If the docstring's Examples block — or the GitHub usage you found — uses \
a specific factory, loader, or required configuration object, mirror that \
idiom rather than substituting a bare default constructor.
6. If the API returns a structured object (wrapper / container / dataclass), \
access the documented field(s) of the result before using it further.
{feedback_block}
Return strict JSON with no prose or markdown:
{{
  "examples": ["<example 1>", "<example 2>"]
}}"""

_FEEDBACK_BLOCK = (
    "\nThe previous attempt's examples were ALL rejected. "
    "Rejection reasons (full error messages):\n"
    "{numbered}\n\n"
    "If a rejection says \"runtime error\" with a traceback that "
    "originates inside the library, the example reached real "
    "library code — it's just missing a required setup step "
    "the docstring does NOT mention (for instance: a mode "
    "toggle, a tiling/slicing call, a specific dtype or device, "
    "an eval() call, or a config object the constructor "
    "needs). Look at the library's test files and example "
    "scripts on GitHub for the minimal working setup, then "
    "return corrected examples that include whatever extra "
    "calls are required before the failing line.\n"
)

# Names that never belong in a user-facing parameter list.
_PARAM_DROP: frozenset[str] = frozenset({"self", "cls", "args", "kwargs", "*args", "**kwargs"})
_WRAPPED_TYPE_RE = re.compile(r"^<(?:class|enum) '([^']+)'>$")
_OBJECT_REPR_RE = re.compile(r"^<.*>$")     # "<factory>", "<x.Y object at 0x...>"


@dataclass
class Surface:
    s_param: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    examples: list[Example] = field(default_factory=list)
    requires_file_io: bool = False

    @property
    def signature(self) -> str:
        parts = []
        for p in self.s_param:
            s = p.name + (f": {p.type}" if p.type else "") + (f" = {p.default}" if p.default else "")
            parts.append(s)
        return "(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# S_param: introspected signature as the skeleton
# ---------------------------------------------------------------------------


def _clean_annotation(ann: str | None) -> str | None:
    """``"<class 'int'>"`` -> ``"int"``; real type expressions are kept."""
    if not ann:
        return None
    m = _WRAPPED_TYPE_RE.match(ann.strip())
    return m.group(1).rsplit(".", 1)[-1] if m else ann


def _from_signature(signature: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """``(defaults, annotations)`` parsed from ``str(inspect.signature(...))`` as source.

    Stage 1 stores each default as ``str(default)`` (``'hann'`` becomes ``hann``); the signature
    string is written with ``repr`` and recovers the literal a caller would type.
    """
    if not signature:
        return {}, {}
    try:
        fn = ast.parse(f"def _sig{signature}:\n    pass").body[0]
    except (SyntaxError, ValueError, IndexError):
        return {}, {}
    a = fn.args
    positional = a.posonlyargs + a.args
    defaults: dict[str, str] = {}
    for arg, d in zip(positional[len(positional) - len(a.defaults):], a.defaults):
        defaults[arg.arg] = ast.unparse(d)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            defaults[arg.arg] = ast.unparse(d)
    annotations = {}
    for arg in positional + a.kwonlyargs:
        if arg.annotation is None:
            continue
        if isinstance(arg.annotation, ast.Constant) and isinstance(arg.annotation.value, str):
            annotations[arg.arg] = arg.annotation.value       # forward reference
        else:
            annotations[arg.arg] = ast.unparse(arg.annotation)
    return defaults, annotations


def reconcile_parameters(entry: APIEntry, llm_params: list[Parameter]) -> list[Parameter]:
    """S_param with the introspected parameters as the skeleton.

    Names and defaults come from the signature; the type from its annotation, else from the
    extraction model; descriptions from the extraction model, matched by name. Without an
    introspectable signature (C extensions) the model's list is kept.
    """
    by_name = {p.name: p for p in llm_params}
    introspected = [p for p in (entry.parameters or []) if p.name not in _PARAM_DROP]
    if not introspected:
        return [p for p in llm_params if p.name not in _PARAM_DROP]
    sig_defaults, sig_annotations = _from_signature(entry.signature)

    def default_for(p) -> str | None:
        if p.name in sig_defaults:
            return sig_defaults[p.name]
        if p.default is not None and not _OBJECT_REPR_RE.match(p.default.strip()):
            return p.default
        llm = by_name.get(p.name)
        return llm.default if llm else None

    out = []
    for p in introspected:
        llm = by_name.get(p.name)
        out.append(Parameter(
            name=p.name,
            type=sig_annotations.get(p.name) or _clean_annotation(p.annotation)
            or (llm.type if llm else None),
            default=default_for(p),
            description=llm.description if llm else "",
            constraints=llm.constraints if llm else None))
    return out


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_json_object(raw: str) -> dict:
    """The outermost ``{...}`` of a (possibly fenced) response, or ``{}``."""
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _normalize_return_type(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return None if not value or value.lower() in ("null", "unknown", "n/a") else value


def _docstring_return_type(docstring: str) -> str | None:
    try:
        import docstring_parser
        parsed = docstring_parser.parse(docstring)
        if parsed.returns and parsed.returns.type_name:
            return parsed.returns.type_name
    except Exception:  # noqa: BLE001
        pass
    return None


def format_param_lines(params: list[Parameter]) -> str:
    lines = []
    for p in params:
        line = f"- {p.name}"
        if p.type:
            line += f": {p.type}"
        if p.default is not None:
            line += f" = {p.default}"
        if p.description:
            line += f" — {p.description[:160]}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_s(entry: APIEntry, version: str, llm: StrongLLM) -> Surface:
    """S_param, return type and the file-I/O flag."""
    docstring = (entry.docstring or "").strip()[:2000]
    prompt = _SURFACE_PROMPT.format(library=entry.library, version=version,
                                    api_name=entry.api_name,
                                    docstring=docstring or "(no docstring available)")
    data = parse_json_object(llm.generate_with_web_search(prompt, system=_SYSTEM, max_tokens=1024))
    llm_params = []
    for p in data.get("parameters") or []:
        if isinstance(p, dict) and p.get("name") and p["name"] not in _PARAM_DROP:
            llm_params.append(Parameter(name=p["name"], type=p.get("type") or None,
                                        default=p.get("default") or None,
                                        description=p.get("description") or ""))
    return_type = (_normalize_return_type(data.get("return_type"))
                   or _docstring_return_type(entry.docstring or ""))
    if return_type is None and entry.kind == "class":
        return_type = entry.api_name
    return Surface(s_param=reconcile_parameters(entry, llm_params), return_type=return_type,
                   requires_file_io=bool(data.get("requires_file_io", False)))


def generate_examples(entry: APIEntry, version: str, surface: Surface, llm: StrongLLM,
                      cfg: DictConfig, env_python: str) -> list[Example]:
    """Up to ``max_examples`` validated examples (executed or static)."""
    ec = cfg.construction.stage2.examples
    pid_ns = bool(cfg.construction.sandbox.pid_namespace)
    docstring = (entry.docstring or "").strip()[:2000]
    feedback, kept = "", []
    for attempt in range(int(ec.max_retries) + 1):
        prompt = _EXAMPLES_PROMPT.format(
            n=int(ec.num_candidates), library=entry.library, version=version,
            api_name=entry.api_name, signature=surface.signature,
            param_lines=format_param_lines(surface.s_param) or "(no parameters documented)",
            docstring=docstring or "(no docstring available)", feedback_block=feedback)
        data = parse_json_object(llm.generate(prompt, system=_SYSTEM, max_tokens=1500))
        rejected = []
        for cand in (c for c in data.get("examples") or [] if isinstance(c, str) and c.strip()):
            code = strip_code_fences(cand)
            v = validate_example(code, entry.api_name, timeout=int(ec.sandbox_timeout_seconds),
                                 max_memory_mb=int(ec.sandbox_memory_mb), env_python=env_python,
                                 pid_namespace=pid_ns)
            if v.valid:
                kept.append(Example(code=code, status=v.status, reason=v.reason))
                if len(kept) >= int(ec.max_examples):
                    break
            else:
                rejected.append(v.reason)
        if kept:
            break
        if attempt < int(ec.max_retries) and rejected:
            numbered = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(rejected[:5]))
            feedback = _FEEDBACK_BLOCK.format(numbered=numbered)
    logger.debug(f"{entry.api_name}: {len(kept)} valid examples")
    return kept[: int(ec.max_examples)]
