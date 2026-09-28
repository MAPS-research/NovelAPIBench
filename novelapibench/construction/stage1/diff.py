"""Version diff of two API maps (Appendix B.2, "Stage 1").

APIs are matched across versions by identity, not by path, because a walk can surface the
same object under a different path in each version: identity is (short name, source-file
basename, kind), or (short name, parameter shape, kind) for objects without Python source.
Unmatched new entries are *newly introduced*, unless one of these says they already existed:

* the exact fully qualified name resolved in the old version (a private implementation file
  was renamed), with the same kind;
* a relocation key matches an old entry: short name + kind + parameter shape + docstring
  prefix; or, tolerating a signature change, short name + kind + a substantial docstring; or
  short name + kind + a parameter list of at least two parameters; or, last, near-identical
  documentation (SequenceMatcher ratio >= 0.92) of an old entry with the same short name and
  kind.

A pre-existing API is *signature-modified* when its parameters were added or removed, or
changed name, kind or default (``self`` ignored; annotation-only changes ignored); otherwise it
is not a candidate. The previous signature is kept for modified APIs.
"""

from __future__ import annotations

import difflib
import os
import re
from collections import defaultdict

from novelapibench.construction.schemas import APIEntry

# Relocation thresholds.
RELOCATION_DOC_MIN = 40        # docstring characters needed for docstring-based matching
RELOCATION_PARAM_MIN = 2       # parameters needed for a shape-only match
RELOCATION_FUZZY_RATIO = 0.92  # docstring similarity of the last-resort match

#: The ``repr`` of a sentinel default embeds its address, which differs on every run.
_OBJ_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
#: Defaults resolved from the machine at introspection time (a path inside the version's own
#: environment, the interpreter, a temporary or per-job scratch directory) differ between the
#: two introspections without any API change.
_ENV_PATH_RE = re.compile(
    r"(/[^\s'\"]*/(?:site-packages|lib-dynload)/[^\s'\"]*"   # inside an environment
    r"|/[^\s'\"]*/bin/python[\d.]*"                          # an environment's interpreter
    r"|/[^\s'\"]*/job-\d+[^\s'\"]*"                          # per-job scratch directory
    r"|/[^\s'\"]*tmp[^\s'\"]*"                               # temporary directories
    r"|/tmp/[A-Za-z0-9_]{6,})"
)


def identity_key(entry: APIEntry) -> tuple:
    short = entry.api_name.rsplit(".", 1)[-1]
    if entry.source_file:
        return (short, os.path.basename(entry.source_file), entry.kind)
    return (short, tuple((p.name, p.kind) for p in entry.parameters), entry.kind, "__no_source__")


def pick_canonical(entries: list[APIEntry]) -> APIEntry:
    """Among entries sharing an identity in one version: shortest path, then alphabetical."""
    return min(entries, key=lambda e: (e.api_name.count("."), e.api_name))


def relocation_key(entry: APIEntry) -> tuple:
    return (entry.api_name.rsplit(".", 1)[-1], entry.kind,
            tuple((p.name, p.kind) for p in entry.parameters),
            (entry.docstring or "").strip()[:400])


def relocation_key_loose(entry: APIEntry) -> tuple | None:
    doc = (entry.docstring or "").strip()
    if len(doc) < RELOCATION_DOC_MIN:
        return None
    return (entry.api_name.rsplit(".", 1)[-1], entry.kind, doc[:400])


def relocation_key_by_shape(entry: APIEntry) -> tuple | None:
    shape = tuple((p.name, p.kind) for p in entry.parameters)
    if len(shape) < RELOCATION_PARAM_MIN:
        return None
    return (entry.api_name.rsplit(".", 1)[-1], entry.kind, shape)


def normalize_default(value: str | None) -> str | None:
    if value is None:
        return None
    return _ENV_PATH_RE.sub("<ENVPATH>", _OBJ_ADDR_RE.sub("0xADDR", value))


def signature_changed(old: APIEntry, new: APIEntry) -> bool:
    """Parameter count, a name, a kind or a default changed (``self`` and annotations ignored).
    An old entry without an inspectable signature is never reported as changed."""
    if not old.parameters and new.parameters:
        return False
    old_params = [p for p in old.parameters if p.name != "self"]
    new_params = [p for p in new.parameters if p.name != "self"]
    if len(old_params) != len(new_params):
        return True
    for op, np_ in zip(old_params, new_params):
        if op.name != np_.name or op.kind != np_.kind:
            return True
        if normalize_default(op.default) != normalize_default(np_.default):
            return True
    return False


def _mark_modified(e: APIEntry, old: APIEntry, **flags) -> None:
    e.is_modified = True
    e.extra["old_signature"] = old.signature
    e.extra["old_parameters"] = [p.model_dump() for p in old.parameters]
    e.extra.update(flags)


def diff_maps(old_map: dict[str, APIEntry], new_map: dict[str, APIEntry], old_version: str,
              new_version: str, domain: str,
              include_modified: bool = True) -> tuple[list[APIEntry], list[APIEntry]]:
    """``(added, modified)`` of ``new_map`` relative to ``old_map``, each sorted by name."""
    old_by: dict[tuple, list[APIEntry]] = defaultdict(list)
    for e in old_map.values():
        old_by[identity_key(e)].append(e)
    new_by: dict[tuple, list[APIEntry]] = defaultdict(list)
    for e in new_map.values():
        new_by[identity_key(e)].append(e)

    old_by_reloc: dict[tuple, APIEntry] = {}
    old_by_reloc_loose: dict[tuple, APIEntry] = {}
    old_by_reloc_shape: dict[tuple, APIEntry] = {}
    old_by_leaf_kind: dict[tuple, list[APIEntry]] = {}
    for e in old_map.values():
        old_by_reloc.setdefault(relocation_key(e), e)
        if (k := relocation_key_loose(e)) is not None:
            old_by_reloc_loose.setdefault(k, e)
        if (k := relocation_key_by_shape(e)) is not None:
            old_by_reloc_shape.setdefault(k, e)
        old_by_leaf_kind.setdefault((e.api_name.rsplit(".", 1)[-1], e.kind), []).append(e)

    added: list[APIEntry] = []
    modified: list[APIEntry] = []
    for ident in new_by.keys() - old_by.keys():
        e = pick_canonical(new_by[ident]).model_copy(deep=True)
        e.old_version, e.new_version, e.domain = old_version, new_version, domain
        prior = old_map.get(e.api_name)
        if prior is not None and prior.kind != e.kind:
            prior = None                     # same path, different kind: a different API
        if prior is None and relocation_key(e) in old_by_reloc:
            continue                         # moved without change
        if prior is None:
            k = relocation_key_loose(e)
            moved = old_by_reloc_loose.get(k) if k else None
            if moved is None:
                k = relocation_key_by_shape(e)
                moved = old_by_reloc_shape.get(k) if k else None
            if moved is None:
                doc = (e.docstring or "").strip()[:400]
                same = old_by_leaf_kind.get((e.api_name.rsplit(".", 1)[-1], e.kind), [])
                if same and len(doc) >= RELOCATION_DOC_MIN:
                    best, best_ratio = None, 0.0
                    for cand in same:
                        cand_doc = (cand.docstring or "").strip()[:400]
                        if len(cand_doc) < RELOCATION_DOC_MIN:
                            continue
                        ratio = difflib.SequenceMatcher(None, doc, cand_doc).ratio()
                        if ratio > best_ratio:
                            best, best_ratio = cand, ratio
                    if best is not None and best_ratio >= RELOCATION_FUZZY_RATIO:
                        moved = best
            if moved is not None:
                # Moved and re-signed: pre-existing, so modified; moved only: not a candidate.
                if include_modified and signature_changed(moved, e):
                    _mark_modified(e, moved, old_api_name=moved.api_name, relocated=True)
                    modified.append(e)
                continue
        if prior is not None:
            if include_modified and signature_changed(prior, e):
                _mark_modified(e, prior, identity_drift=True)
                modified.append(e)
            continue
        e.is_modified = False
        added.append(e)

    if include_modified:
        for ident in new_by.keys() & old_by.keys():
            oe, ne = pick_canonical(old_by[ident]), pick_canonical(new_by[ident])
            if signature_changed(oe, ne):
                e = ne.model_copy(deep=True)
                e.old_version, e.new_version, e.domain = old_version, new_version, domain
                _mark_modified(e, oe)
                if oe.api_name != ne.api_name:
                    e.extra["old_api_name"] = oe.api_name
                modified.append(e)

    added.sort(key=lambda e: e.api_name)
    modified.sort(key=lambda e: e.api_name)
    return added, modified
