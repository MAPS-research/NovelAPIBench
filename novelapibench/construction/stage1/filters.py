"""Candidate filters (Appendix B.2, "Stage 1").

Per-entry filters (each judges one entry on its own, so applying them per boundary or on the
union gives the same answer) remove APIs that are private, deprecated, typing/stdlib
re-exports, undocumented or thinly documented (< 40 characters, or a docstring inherited from
a base class or builtin), without Python source, defined outside the library, cross-cutting
private plumbing re-exported under an unrelated public path, vendored third-party code,
self-declared private or internal, or defined in a module named ``base``, ``main``, ``impl``
or ``internals``.

Set-level selection depends on the size of the candidate set and is therefore applied once,
on the union over boundaries: the CamelCase-suffix cap (at most three classes from any group
of at least five new classes sharing a suffix, by quality) and the per-library candidate cap
(maximal marginal relevance over MiniLM embeddings of name and docstring, lambda = 0.5).
"""

from __future__ import annotations

import collections
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from novelapibench.construction.schemas import APIEntry
from novelapibench.log import logger

_DEPRECATED_PATTERNS = [
    re.compile(r"\bdeprecated\b", re.IGNORECASE),
    re.compile(r"will be removed", re.IGNORECASE),
    re.compile(r"use .+ instead", re.IGNORECASE),
]

# typing / collections.abc / datetime names that libraries re-export.
_TYPING_REEXPORT_NAMES: frozenset[str] = frozenset({
    "Any", "Callable", "ClassVar", "Dict", "FrozenSet", "Generator", "Generic", "Iterable",
    "Iterator", "List", "Literal", "Mapping", "NamedTuple", "Optional", "Protocol", "Sequence",
    "Set", "Tuple", "Type", "TypedDict", "TypeVar", "Union",
    "timedelta", "datetime", "date", "time", "timezone",
})
_UNINFORMATIVE_PARAM_NAMES: frozenset[str] = frozenset({"self", "cls", "args", "kwargs"})

# Opening words (of the whitespace-collapsed first paragraph) of docstrings that belong to a
# base class or builtin, or carry no documentation.
_INHERITED_DOCSTRING_PREFIXES = (
    "helper class that provides a standard way to create an abc",   # abc.ABC
    "implement setattr(self, name, value)",                          # object
    "implement getattr(self, name)",                                 # object
    "abstract base class for",
    "thanks to chatgpt",
    "todo",
    "fixme",
    "xxx:",
    "note:",
    "base class for protocol classes",                               # typing.Protocol
    "dict() -> new empty dictionary",                                # dict / TypedDict
    "str(object=",                                                   # str / str-Enum
    "common base class for all non-exit exceptions",                 # Exception
    "common base class for all exceptions",                          # BaseException
    "information about how to convert command line strings",         # argparse.Action
    "base class for all neural network modules",                     # torch.nn.Module
)

# Package directories holding cross-cutting plumbing rather than one public namespace's
# implementation (``_core`` and ``_src`` are private implementations OF a namespace: kept).
_PLUMBING_DIR_SEGMENTS = frozenset({"_lib", "_internal", "_compat", "_utils", "_vendor",
                                    "_vendored", "array_api_compat"})
# Subpackages that vendor a third-party project verbatim (pandas.io.clipboard is pyperclip).
_VENDORED_SUBPACKAGES = ("pandas.io.clipboard",)
_SELF_DECLARED_PRIVATE_RE = re.compile(r"^\s*(private|internal(?:\s+use)?(?:\s+only)?)\b",
                                       re.IGNORECASE)
_COMPILED_EXTENSIONS = (".so", ".pyd", ".dll")
DEFAULT_INTERNAL_MODULE_SEGMENTS = ("base", "main", "impl", "internals")


# ---------------------------------------------------------------------------
# Per-entry predicates
# ---------------------------------------------------------------------------


def quality_score(entry: APIEntry) -> float:
    """Docstring length times the number of informative parameters (at least 1)."""
    n_params = sum(1 for p in entry.parameters if p.name not in _UNINFORMATIVE_PARAM_NAMES)
    return len(entry.docstring or "") * max(1, n_params)


def top_by_quality(entries: list[APIEntry], n: int) -> list[APIEntry]:
    return sorted(entries, key=quality_score, reverse=True)[:n]


def is_private(entry: APIEntry) -> bool:
    return entry.api_name.split(".")[-1].startswith("_")


def is_deprecated(entry: APIEntry) -> bool:
    return any(p.search(entry.docstring or "") for p in _DEPRECATED_PATTERNS)


def is_type_alias(entry: APIEntry) -> bool:
    if entry.kind in ("function", "method") and not entry.parameters and not entry.docstring:
        return True
    return entry.kind == "class" and entry.api_name.split(".")[-1] in _TYPING_REEXPORT_NAMES


def is_docstring_too_thin(entry: APIEntry, min_chars: int = 40) -> bool:
    doc = (entry.docstring or "").strip()
    if len(doc) < min_chars:
        return True
    first_para = " ".join(doc.split("\n\n", 1)[0].split()).lower()
    return any(first_para.startswith(p) for p in _INHERITED_DOCSTRING_PREFIXES)


def has_no_source(entry: APIEntry) -> bool:
    """C-implemented or compiled-extension objects (no implementation component C)."""
    return entry.source_file is None or entry.source_file.lower().endswith(_COMPILED_EXTENSIONS)


def is_unrelated_plumbing(entry: APIEntry) -> bool:
    """Defined in a plumbing directory that shares no namespace segment with the public path."""
    source = entry.source_file or ""
    idx = source.find("site-packages/")
    if idx < 0:
        return False
    rel_dirs = source[idx + len("site-packages/"):].split("/")[:-1]
    if not set(rel_dirs) & _PLUMBING_DIR_SEGMENTS:
        return False
    public = entry.api_name.split(".")[:-1]
    if any(p.startswith("_") for p in public):
        return False
    return not (set(rel_dirs[1:]) & set(public[1:]))


def is_vendored_subpackage(entry: APIEntry) -> bool:
    return any(entry.api_name.startswith(v + ".") for v in _VENDORED_SUBPACKAGES)


def is_self_declared_private(entry: APIEntry) -> bool:
    return bool(_SELF_DECLARED_PRIVATE_RE.match(entry.docstring or ""))


def is_foreign_source(entry: APIEntry, package: str) -> bool:
    """The source file does not live in the library's package directory."""
    if not package or entry.source_file is None:
        return False
    return package not in entry.source_file.replace("\\", "/").split("/")


def has_internal_module_leaf(entry: APIEntry,
                             segments: tuple[str, ...] = DEFAULT_INTERNAL_MODULE_SEGMENTS) -> bool:
    parts = entry.api_name.split(".")
    return len(parts) >= 3 and parts[-2] in segments


def apply_filters(entries: list[APIEntry], package: str, min_docstring_chars: int = 40,
                  internal_module_segments: tuple[str, ...] = DEFAULT_INTERNAL_MODULE_SEGMENTS,
                  exclude_name_patterns: list[str] | None = None) -> list[APIEntry]:
    """The per-entry filters, in order; the input order is preserved."""
    patterns = [re.compile(p) for p in exclude_name_patterns or []]
    kept = []
    for e in entries:
        if (is_private(e) or is_deprecated(e) or is_type_alias(e) or not e.docstring
                or is_docstring_too_thin(e, min_docstring_chars) or has_no_source(e)
                or is_unrelated_plumbing(e) or is_vendored_subpackage(e)
                or is_self_declared_private(e) or is_foreign_source(e, package)
                or has_internal_module_leaf(e, tuple(internal_module_segments))
                or any(p.search(e.api_name.split(".")[-1]) for p in patterns)):
            continue
        kept.append(e)
    return kept


def boilerplate_docstrings(api_map: dict[str, APIEntry], min_shared: int = 3) -> set[str]:
    """Docstring prefixes shared by ``min_shared`` or more APIs of one library version: they are
    inherited from a base class, not documentation of any one API."""
    counts: collections.Counter = collections.Counter()
    for e in api_map.values():
        doc = (e.docstring or "").strip()[:400]
        if doc:
            counts[doc] += 1
    return {d for d, n in counts.items() if n >= min_shared}


def under_failed_subtree(api_name: str, failures: set[str]) -> bool:
    return any(api_name == s or api_name.startswith(s + ".") for s in failures)


# ---------------------------------------------------------------------------
# Set-level selection
# ---------------------------------------------------------------------------


def _camel_suffix(name: str) -> str:
    """Last CamelCase word of at least three letters (``LlamaForCausalLM`` -> ``Causal``)."""
    meaningful = [p for p in re.findall(r"[A-Z][a-z]+", name) if len(p) >= 3]
    return meaningful[-1] if meaningful else name


def apply_suffix_cap(entries: list[APIEntry], min_group_size: int = 5,
                     max_per_suffix: int = 3) -> tuple[list[APIEntry], dict[str, int]]:
    """Keep the ``max_per_suffix`` best new classes of each suffix group of at least
    ``min_group_size``; functions and modified entries pass through. Order is preserved."""
    groupable = {e.api_name for e in entries if e.kind == "class" and not e.is_modified}
    by_suffix: dict[str, list[APIEntry]] = defaultdict(list)
    for e in entries:
        if e.api_name in groupable:
            by_suffix[_camel_suffix(e.api_name.split(".")[-1])].append(e)
    kept: set[str] = {e.api_name for e in entries if e.api_name not in groupable}
    capped: dict[str, int] = {}
    for suffix, group in by_suffix.items():
        if len(group) >= min_group_size:
            chosen = top_by_quality(group, max_per_suffix)
            capped[suffix] = len(group) - len(chosen)
            kept.update(e.api_name for e in chosen)
        else:
            kept.update(e.api_name for e in group)
    return [e for e in entries if e.api_name in kept], capped


@lru_cache(maxsize=1)
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def mmr_select(entries: list[APIEntry], n: int, lambda_: float = 0.5,
               model_name: str = "all-MiniLM-L6-v2", max_doc_chars: int = 500) -> list[APIEntry]:
    """Maximal marginal relevance (Carbonell & Goldstein, 1998).

    Greedily picks ``argmax  lambda * quality(e) - (1 - lambda) * max_s cos(e, s)`` over the
    remaining entries, starting from the best-quality one; quality is min-max normalised and
    embeddings are L2-normalised sentence embeddings of ``"<name>: <docstring[:500]>"``.
    """
    if len(entries) <= n:
        return entries
    texts = [f"{e.api_name}: {(e.docstring or '')[:max_doc_chars]}" for e in entries]
    logger.info(f"MMR: encoding {len(texts)} entries with {model_name}")
    embs = _encoder(model_name).encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                       batch_size=64)
    raw_q = np.array([quality_score(e) for e in entries], dtype=float)
    lo, hi = float(raw_q.min()), float(raw_q.max())
    q = np.ones_like(raw_q) if hi == lo else (raw_q - lo) / (hi - lo)

    selected = [int(np.argmax(q))]
    remaining = [i for i in range(len(entries)) if i != selected[0]]
    while len(selected) < n and remaining:
        max_sim = (embs[remaining] @ embs[selected].T).max(axis=1)
        scores = lambda_ * q[remaining] - (1.0 - lambda_) * max_sim
        best = int(np.argmax(scores))
        selected.append(remaining.pop(best))
    return [entries[i] for i in selected]
