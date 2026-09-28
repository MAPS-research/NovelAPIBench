"""Mechanism description M (Appendix B.2, "Stage 2").

M is grounded in one of three sources, chosen per API:

* ``paper``: the docstring cites a paper (arXiv / DOI / author-year), or, for an API with
  substantial source, a web search finds the paper the documentation attributes the component
  to. Two steps: GPT-5-mini with web search identifies the paper(s) and how the component
  relates to them (a concrete implementation, a container of several methods, abstract
  scaffolding); then a 200-400-word explanation of the component is written with the papers
  as background. Scaffolding, no paper, or a hedging answer falls back to ``source``.
* ``source``: functions with a long docstring and plain classes whose implementation C has at
  least 25 non-comment lines: a 200-400-word explanation of the algorithm and design, written
  from C.
* ``docstring``: everything else, including structural classes (exceptions, enums, protocols,
  typed dicts, named tuples, dataclasses): a 50-100-word description from the docstring.

Prompts are specialised by API role (function, or the class's role). An answer that opens
with a stock phrase ("This function ...") is regenerated once. Changelog lines that mention
the API are attached as notes.

The prompts are the ones used to build the released benchmark.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import requests
from omegaconf import DictConfig

from novelapibench.construction.schemas import APIEntry
from novelapibench.llm.strong import StrongLLM
from novelapibench.schemas import Mechanism

_SYSTEM = (
    "You are an expert technical writer. Explain scientific and algorithmic concepts "
    "clearly and precisely, prioritising conceptual understanding over API surface "
    "detail. Referencing API or parameter names is fine when it genuinely aids the "
    "explanation."
)

# Shared by the source- and docstring-grounded prompts.
_ANTI_BOILERPLATE_CLAUSE = (
    "Do NOT open with 'This {role}...', 'The {role} in question...', "
    "'This class is designed to...', 'This function performs...'. Start the "
    "first sentence directly with the concrete behaviour, invariant, or "
    "domain concept."
)

# Openings that trigger one retry (matched on the lowercased first line).
_BOILERPLATE_OPENINGS = (
    "this function ",
    "the function ",
    "this class ",
    "the class ",
    "this module ",
    "this component ",
    "the component ",
    "this utility ",
    "the utility ",
    "this exception ",
    "the exception ",
    "this enum ",
    "the enum ",
    "this protocol ",
    "the protocol ",
    "this dataclass ",
    "this named tuple ",
    "this namedtuple ",
    "this typed dict ",
    "this typeddict ",
)

# Source-grounded prompts (200-400 words), by API role.
_SOURCE_TASK = {
    "function": (
        "Explain:\n"
        "1. The underlying algorithm or computational approach.\n"
        "2. The domain knowledge required to use this correctly.\n"
        "3. Key constraints and assumptions.\n"
        "4. When this function is appropriate to use."
    ),
    "exception": (
        "Describe:\n"
        "1. The error condition this exception signals — what contract "
        "violation or runtime state triggers it.\n"
        "2. Which library operations raise it, and under what inputs.\n"
        "3. What a caller should do in response (recover, re-raise, "
        "reconfigure, re-try with different inputs).\n"
        "Do NOT describe this as a parser, validator, or transformer — "
        "it does not compute a return value; it signals failure."
    ),
    "enum": (
        "Describe:\n"
        "1. The closed set of named values this enum defines and the "
        "domain concept they enumerate.\n"
        "2. What each member means and the situations in which a caller "
        "chooses that member.\n"
        "3. How this enum is consumed by other APIs in the library.\n"
        "Do NOT describe this as a function or an operation — it is a "
        "discrete set of labels."
    ),
    "protocol": (
        "Describe:\n"
        "1. The structural interface this Protocol defines — which "
        "methods and attributes a conforming type must provide.\n"
        "2. The invariants those methods are expected to uphold.\n"
        "3. Which concrete types in the library or ecosystem satisfy it.\n"
        "Emphasise that this is a typing contract, not an implementation. "
        "Do NOT summarise PEP 544 / structural subtyping in general — "
        "describe THIS protocol's contract."
    ),
    "dataclass": (
        "Describe:\n"
        "1. The structured record this dataclass represents — what "
        "domain entity its fields collectively model.\n"
        "2. Any validation or normalisation performed in ``__post_init__`` "
        "or field validators.\n"
        "3. Where in the library it is constructed and where it is "
        "consumed.\n"
        "Treat this as a data shape, not an operation."
    ),
    "namedtuple": (
        "Describe:\n"
        "1. The immutable tuple-shaped record — what the positional "
        "fields represent.\n"
        "2. Why tuple semantics were chosen (hashability, pattern "
        "matching, legacy API compatibility).\n"
        "3. Which APIs return values of this shape and how callers "
        "typically unpack them."
    ),
    "typeddict": (
        "Describe:\n"
        "1. The dict schema this TypedDict defines — which keys are "
        "required versus optional, and what values each key carries.\n"
        "2. Which library APIs accept or return dicts matching this "
        "shape.\n"
        "3. Any cross-key constraints the library expects callers to "
        "maintain."
    ),
    "plain_class": (
        "Describe:\n"
        "1. What instances of this class represent or manage (state, "
        "resource, coordinator, builder, configuration).\n"
        "2. The instance lifecycle — how it is constructed, how it is "
        "used, and whether it must be disposed or closed.\n"
        "3. Invariants the class maintains and its role in the library's "
        "architecture.\n"
        "Do NOT describe this as a function."
    ),
}

_SOURCE_PROMPT = """\
The following is the source code of this {role}:

```python
{source_code}
```

Changelog context (if available):
{changelog}

{core_task}

Keep the focus on concepts and mechanisms rather than a parameter-by-parameter
walkthrough. Write in 200-400 words.

{anti_boilerplate}
"""

# Docstring-grounded prompts (50-100 words), by API role.
_DOCSTRING_TASK = {
    "function": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. What this function does and what it returns.\n"
        "2. The typical use case or when to call it."
    ),
    "exception": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. What failure condition this exception signals and when it is "
        "raised.\n"
        "2. How callers typically handle it.\n"
        "Do NOT describe this as a function."
    ),
    "enum": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. The discrete states this enum defines.\n"
        "2. How callers choose between the members."
    ),
    "protocol": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. The structural interface this Protocol requires of conforming "
        "types.\n"
        "2. What concrete types typically satisfy it."
    ),
    "dataclass": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. The structured record this dataclass represents and its "
        "fields.\n"
        "2. Where it is produced or consumed."
    ),
    "namedtuple": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. The immutable record this named tuple represents and its "
        "fields.\n"
        "2. Where it is returned from."
    ),
    "typeddict": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. The dict schema: required versus optional keys and value "
        "types.\n"
        "2. Which APIs accept or return dicts of this shape."
    ),
    "plain_class": (
        "Write a concise factual description (50-100 words) covering:\n"
        "1. What instances of this class represent or manage.\n"
        "2. The typical use case or lifecycle."
    ),
}

_DOCSTRING_PROMPT = """\
The following is the documentation for a library {role}:

```
{docstring}
```

{core_task}

Be factual and concrete; keep the focus on what it is conceptually
rather than an exhaustive parameter-by-parameter walkthrough.

{anti_boilerplate}
"""

# Paper-grounded prompts: identify the paper (web search), then explain the component.
_PAPER_IDENTIFY_PROMPT = """\
Identify the specific research paper(s) that the `{api_leaf}` component from \
the `{library}` Python library is actually based on.

Steps:
1. Look up the official documentation page for `{api_leaf}` on {doc_url} \
(or search "{api_leaf} {library} docs"). The docs usually name the original \
paper, arXiv ID, or model family the component implements.
2. If the docs name a paper, verify its title with a second search.
3. Decide whether that paper IS the core mechanism, or just ONE option the \
component supports, or unrelated.

Brief description from documentation: {description}
{citation_hint}

Classify the paper-component relationship into one of:
- "concrete_impl": Implements a specific published method; the paper is the \
core mechanism.
- "generic_container": A framework / mixin / dispatcher that SUPPORTS several \
methods (possibly backed by different papers).
- "abstract_base": An abstract base class, registry, config holder, or \
generic framework scaffolding whose behaviour is defined by engineering \
choices, not a paper.
- "unknown": Cannot tell from available sources.

List ONLY papers that the documentation or source directly attributes to \
this component. Do NOT list topically-related papers from the broader field. \
If there is no such paper, return an empty list.

Respond with ONLY a JSON object, no prose, no markdown fences, matching \
exactly this schema:
{{"paper_relation": "concrete_impl" | "generic_container" | "abstract_base" | "unknown",
 "papers": [{{"title": "<title>", "arxiv_id": "<id or null>", "role": "core" | "one_of_many"}}],
 "notes": "<1-2 sentences summarising what the docs say about the paper-component link>"}}
"""

_PAPER_EXPLAIN_PROMPT = """\
Explain the underlying principle of a component from the `{library}` Python \
library. The paper(s) backing this component have already been identified \
for you — do NOT summarise those papers in isolation; use them only as \
background for explaining the component.

Component role in library: {api_role}
Paper relation: {paper_relation}
Documentation description: {description}
Identified paper(s):
{papers_block}
Notes on the paper-component relationship: {notes}

Cover, in 200-400 words, all centred on THIS component:
1. The problem the component addresses and its high-level approach.
2. The mathematical or algorithmic core it actually implements. If the \
component is a generic_container, describe the shared abstraction it exposes \
and briefly note that specific backends correspond to specific papers.
3. The key design decisions and their rationale.
4. When to use it vs. alternatives; important constraints or assumptions.

Rules:
- Do NOT write a "Related work" / "In the broader context" / "Notable papers \
in this area" section.
- Do NOT spend more than 2-3 sentences on any single paper's isolated \
history, authors, or experiments — always redirect to how the component \
uses that idea.
- Do NOT hedge with phrases like "there is no direct reference" or \
"appears to be a foundational framework" — if that were true, you would \
not have been called.
- Do NOT open with a boilerplate meta-phrase such as "The component in \
question...", "This component addresses...", "The component addresses \
the challenge of...", "This module is designed to...". Start the first \
sentence directly with the concrete problem domain, mathematical idea, \
or mechanism itself (e.g. "Latent-space diffusion reduces the compute \
cost of...", "Discrete token prediction over a learned codebook..."). \
Vary the opening — do not reuse the same stock phrasing across responses.
"""

# An explanation containing one of these hedges is not paper-grounded.
_NO_PAPER_HEDGE_PHRASES = (
    "no direct reference to a specific research paper",
    "no specific research paper",
    "no specific paper",
    "not directly referenced",
    "does not reference a specific paper",
    "not based on a singular research work",
    "foundational framework within the library",
    "in the broader context",
    "notable among these are",
    "related work",
)

_ROLES = ("function", "exception", "enum", "protocol", "dataclass", "namedtuple", "typeddict",
          "plain_class")
# Class roles that are definitions, not algorithms: always docstring-grounded.
_STRUCTURAL_ROLES = frozenset({"exception", "enum", "protocol", "typeddict", "namedtuple",
                               "dataclass"})

_ARXIV_RE = re.compile(r"(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_HF_PAPER_RE = re.compile(r"huggingface\.co/papers/(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_DOI_RE = re.compile(r"(?:doi:|https?://doi\.org/)(10\.\d{4,}/[^\s,)\]>\"']+)", re.IGNORECASE)
_AUTHOR_YEAR_RE = re.compile(r"[\[(]([A-Z][a-zA-Z\-]+(?: et al\.?)?,?\s*\d{4})[\])]")


def api_role(entry: APIEntry) -> str:
    if entry.kind == "class":
        role = (entry.extra or {}).get("class_role", "plain_class")
        return role if role in _ROLES else "plain_class"
    return "function"


def docstring_citations(docstring: str) -> list[dict]:
    """arXiv ids (incl. huggingface.co/papers links), DOIs and author-year citations."""
    out, seen = [], set()
    for kind, regex in (("arxiv", _ARXIV_RE), ("arxiv", _HF_PAPER_RE), ("doi", _DOI_RE),
                        ("author_year", _AUTHOR_YEAR_RE)):
        for m in regex.finditer(docstring):
            val = m.group(1).rstrip(".") if kind == "doi" else m.group(1)
            if val not in seen:
                out.append({"type": kind, "value": val})
                seen.add(val)
    return out


def _non_comment_lines(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def _source_lines_proxy(entry: APIEntry) -> int:
    """Implementation size estimated from the docstring length."""
    n = len(entry.docstring or "")
    return 50 if n > 500 else 25 if n > 200 else 15 if n > 50 else 5


def choose_grounding(entry: APIEntry, cfg: DictConfig,
                     implementation: Callable[[], str | None]) -> tuple[str, list[dict]]:
    """``(grounding, citations)``: ``paper`` with docstring citations, ``source`` or ``docstring``."""
    mc = cfg.construction.stage2.mechanism
    if entry.docstring:
        citations = docstring_citations(entry.docstring)
        if citations:
            return "paper", citations
    role = api_role(entry)
    if role in _STRUCTURAL_ROLES:
        return "docstring", []
    if role == "plain_class":
        if len(entry.docstring or "") < 200:
            return "docstring", []
        code = implementation()
        if code and _non_comment_lines(code) >= int(mc.min_implementation_lines):
            return "source", []
        return "docstring", []
    if entry.source_file and _source_lines_proxy(entry) >= int(mc.min_source_lines):
        return "source", []
    return "docstring", []


def _starts_with_boilerplate(text: str) -> bool:
    for line in (text or "").strip().splitlines():
        if line.strip():
            return any(line.strip().lower().startswith(p) for p in _BOILERPLATE_OPENINGS)
    return False


def _generate_with_retry(llm: StrongLLM, prompt: str, role: str) -> str:
    """Generate; if the answer opens with a stock phrase, ask once for a rewrite."""
    text = llm.generate(prompt, system=_SYSTEM)
    if not _starts_with_boilerplate(text):
        return text
    retry = (prompt + "\n\nYour previous response opened with a forbidden "
             "meta-phrase (e.g. 'This {role}...', 'The {role} in "
             "question...'). Rewrite starting directly with the concrete "
             "idea — no meta-phrases about 'this {role}'.".format(role=role))
    try:
        again = llm.generate(retry, system=_SYSTEM)
    except Exception:  # noqa: BLE001
        return text
    return again if again and again.strip() else text


def _parse_identify_json(text: str) -> dict | None:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except Exception:  # noqa: BLE001
        return None


def paper_grounded(entry: APIEntry, doc_url: str, llm: StrongLLM, role: str,
                   changelog_notes: str | None, citations: list[dict] | None = None
                   ) -> Mechanism | None:
    """Identify the paper(s) with web search, then explain the component; None if no paper."""
    leaf = entry.api_name.split(".")[-1]
    doc_snippet = (entry.docstring or "").strip()[:500]
    hint = ""
    for c in (citations or [])[:3]:
        label = {"arxiv": "Known arXiv ID", "doi": "Known DOI", "author_year": "Citation"}.get(c.get("type"))
        if label:
            hint += f"\n{label}: {c.get('value', '')}"
    prompt = _PAPER_IDENTIFY_PROMPT.format(
        api_leaf=leaf, library=entry.library,
        description=doc_snippet or "(No description available)",
        citation_hint=f"\nCitation hints:{hint}" if hint else "",
        doc_url=doc_url or f"the official {entry.library} documentation")
    try:
        parsed = _parse_identify_json(llm.generate_with_web_search(prompt, system=_SYSTEM))
    except Exception:  # noqa: BLE001
        return None
    if parsed is None:
        return None
    relation = str(parsed.get("paper_relation", "unknown")).strip()
    if not relation or relation == "unknown":
        relation = str(parsed.get("component_kind", "")).strip() or relation
    papers = []
    for p in parsed.get("papers") or []:
        if isinstance(p, dict) and str(p.get("title", "")).strip():
            papers.append({"title": str(p["title"]).strip(),
                           "arxiv_id": (p.get("arxiv_id") or "") or None,
                           "role": str(p.get("role", "core")).strip() or "core"})
    if not papers or relation in ("abstract_base", "unknown"):
        return None
    lines = []
    for p in papers:
        line = f"- {p['title']} (role={p['role']})"
        if p["arxiv_id"]:
            line += f" [arXiv:{p['arxiv_id']}]"
        lines.append(line)
    explain = _PAPER_EXPLAIN_PROMPT.format(
        library=entry.library, api_role=role, paper_relation=relation,
        description=doc_snippet or "(No description available)", papers_block="\n".join(lines),
        notes=str(parsed.get("notes", "")).strip() or "(none)")
    try:
        text = llm.generate(explain, system=_SYSTEM)
    except Exception:  # noqa: BLE001
        return None
    if not text or not text.strip() or any(h in text.lower() for h in _NO_PAPER_HEDGE_PHRASES):
        return None
    return Mechanism(text=text.strip(), grounding="paper", references=[p["title"] for p in papers],
                     changelog_notes=changelog_notes)


def source_grounded(entry: APIEntry, llm: StrongLLM, role: str, changelog_notes: str | None,
                    implementation: Callable[[], str | None], max_input_tokens: int) -> Mechanism:
    code = implementation()
    unavailable = not (code and code.strip())
    if unavailable:
        doc = (entry.docstring or "").strip()
        if len(doc) < 200:
            return docstring_grounded(entry, llm, role, changelog_notes)
        context = (doc + "\n\n(Python source unavailable — C-extension or compiled "
                   "implementation; the above is the documentation.)")
    else:
        context = code.strip()
    max_chars = max_input_tokens * 4
    if len(context) > max_chars:
        context = context[:max_chars] + "\n... (truncated)"
    prompt = _SOURCE_PROMPT.format(role=role, source_code=context,
                                   changelog=changelog_notes or "(No changelog context)",
                                   core_task=_SOURCE_TASK.get(role, _SOURCE_TASK["function"]),
                                   anti_boilerplate=_ANTI_BOILERPLATE_CLAUSE.format(role=role))
    return Mechanism(text=_generate_with_retry(llm, prompt, role), grounding="source",
                     changelog_notes=changelog_notes,
                     source_summary="source_unavailable" if unavailable else None)


def docstring_grounded(entry: APIEntry, llm: StrongLLM, role: str,
                       changelog_notes: str | None) -> Mechanism:
    prompt = _DOCSTRING_PROMPT.format(
        role=role, docstring=(entry.docstring or "(No documentation available)")[:2000],
        core_task=_DOCSTRING_TASK.get(role, _DOCSTRING_TASK["function"]),
        anti_boilerplate=_ANTI_BOILERPLATE_CLAUSE.format(role=role))
    return Mechanism(text=_generate_with_retry(llm, prompt, role), grounding="docstring",
                     changelog_notes=changelog_notes)


def extract_mechanism(entry: APIEntry, doc_url: str, cfg: DictConfig, llm: StrongLLM,
                      implementation: Callable[[], str | None],
                      changelog_notes: str | None) -> Mechanism:
    """M for one API (``implementation`` returns C, computed at most once)."""
    mc = cfg.construction.stage2.mechanism
    role = api_role(entry)
    grounding, citations = choose_grounding(entry, cfg, implementation)
    max_tokens = int(mc.max_input_tokens)
    if grounding == "paper":
        m = paper_grounded(entry, doc_url, llm, role, changelog_notes, citations)
        return m or source_grounded(entry, llm, role, changelog_notes, lambda: None, max_tokens)
    if grounding == "source":
        if role in ("function", "plain_class") and mc.paper_web_search:
            m = paper_grounded(entry, doc_url, llm, role, changelog_notes)
            if m is not None:
                return m
        return source_grounded(entry, llm, role, changelog_notes, implementation, max_tokens)
    return docstring_grounded(entry, llm, role, changelog_notes)


# ---------------------------------------------------------------------------
# Changelog notes
# ---------------------------------------------------------------------------


def fetch_changelog(url: str, timeout: int = 10) -> str | None:
    """The raw text of the library's changelog / release-notes page (None on failure)."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return None


def changelog_notes_for(changelog: str | None, api_name: str) -> str | None:
    """Lines (+-3) mentioning the API's name (or a leaf longer than four characters)."""
    if not changelog or not api_name:
        return None
    leaf = api_name.split(".")[-1]
    lines, out = changelog.split("\n"), []
    for i, line in enumerate(lines):
        if api_name in line or (leaf in line and len(leaf) > 4):
            out.extend(lines[max(0, i - 3):min(len(lines), i + 4)])
            out.append("")
    if not out:
        return None
    text = re.sub(r"<[^>]+>", "", "\n".join(out).strip())
    return text[:1000] if text else None
