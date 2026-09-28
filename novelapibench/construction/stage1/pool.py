"""The frozen candidate pool (Appendix B.2, "Stage 1").

For each backbone's release boundary, ``old_version`` is the last PyPI release before the
boundary (``configs/construction.yaml``; ``resolve_old_versions`` recomputes it). A library
diff against each boundary keeps the added and modified APIs that survive the per-entry
filters, minus added APIs under a subtree the old version's walk failed to import (they would
read as new only because they were not seen) and entries whose docstring is boilerplate shared
by three or more APIs of ``new_version``. A library without any release before a boundary
contributes all its filtered public APIs for that boundary.

The pool is the union over boundaries (the first boundary that finds an API supplies its
record; the primary boundary, 2023-12-01, comes first), after which the set-size-dependent
selection (suffix cap, candidate cap with MMR) runs once per library. ``membership.csv``
records, for every candidate and boundary, whether it is ``added`` or ``modified`` relative to
that boundary; instances are not filtered by it (C2 makes the benchmark model-specific).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import urllib.request
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from novelapibench.config import load_library_config
from novelapibench.construction.schemas import APIEntry
from novelapibench.construction.stage1.api_maps import (
    api_map_path, file_sha256, load_api_map, load_map_health)
from novelapibench.construction.stage1.diff import diff_maps
from novelapibench.construction.stage1.filters import (
    apply_filters, apply_suffix_cap, boilerplate_docstrings, mmr_select, top_by_quality,
    under_failed_subtree)
from novelapibench.io import write_json
from novelapibench.log import logger

# Absolute paths into an environment's site-packages are machine-specific.
_SITE_PACKAGES_RE = re.compile(r"[\w./~-]*/lib/python3\.\d+/site-packages/")


def boundaries(cfg: DictConfig) -> dict[str, str]:
    """``backbone -> boundary date``, primary boundary first."""
    return dict(cfg.construction.stage1.boundaries)


def library_old_versions(library: str, cfg: DictConfig) -> dict[str, str | None]:
    """``backbone -> old_version`` (``None``: no release before that boundary)."""
    table = cfg.construction.stage1.old_versions.get(library)
    if table is None:
        raise KeyError(f"no old_versions for {library} in configs/construction.yaml; "
                       "run `build_benchmark.py pool --resolve-boundaries`")
    return {m: (None if table.get(m) is None else str(table.get(m))) for m in boundaries(cfg)}


def resolve_old_versions(package: str, bounds: dict[str, str]) -> dict[str, str | None]:
    """Last stable PyPI release uploaded before each boundary date."""
    req = urllib.request.Request(f"https://pypi.org/pypi/{package}/json",
                                 headers={"User-Agent": "novelapibench-construction/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        releases = json.loads(r.read().decode()).get("releases") or {}

    def key(v: str) -> tuple:
        return tuple(int("".join(c for c in ch if c.isdigit()) or 0)
                     for ch in v.replace("-", ".").split("."))

    def prerelease(v: str) -> bool:
        low = v.lower()
        return (any(t in low for t in ("a", "b", "rc", "dev", "post"))
                and not low.replace(".", "").isdigit())

    uploaded: dict[str, dt.datetime] = {}
    for v, files in releases.items():
        if not files or prerelease(v):
            continue
        stamps = []
        for f in files:
            ts = f.get("upload_time_iso_8601") or f.get("upload_time") or ""
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
                try:
                    stamps.append(dt.datetime.strptime(ts, fmt).replace(tzinfo=dt.timezone.utc))
                    break
                except ValueError:
                    continue
        if stamps:
            uploaded[v] = min(stamps)
    out = {}
    for model, date in bounds.items():
        cut = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
        before = [v for v, t in uploaded.items() if t < cut]
        out[model] = max(before, key=key) if before else None
    return out


def required_versions(library: str, cfg: DictConfig) -> list[str]:
    """Every version the library's API maps are needed for (``new_version`` first)."""
    lib = load_library_config(library)
    versions = [str(lib.new_version)]
    for v in library_old_versions(library, cfg).values():
        if v is not None and v not in versions:
            versions.append(v)
    return versions


# ---------------------------------------------------------------------------
# Per-library diff, union and selection
# ---------------------------------------------------------------------------


def entry_filters(entries: list[APIEntry], library: str, cfg: DictConfig) -> list[APIEntry]:
    lib = load_library_config(library)
    fc = cfg.construction.stage1.filters
    return apply_filters(
        entries, package=str(lib.get("import_name") or lib.package),
        min_docstring_chars=int(fc.min_docstring_chars),
        internal_module_segments=tuple(fc.internal_module_segments),
        exclude_name_patterns=list(fc.get("exclude_name_patterns") or []) +
        list(lib.get("construction", {}).get("exclude_name_patterns") or []))


def max_novel(library: str, cfg: DictConfig) -> int:
    lib = load_library_config(library)
    return int(lib.get("construction", {}).get("max_novel", 0) or cfg.construction.stage1.max_novel)


def union_set_filters(entries: list[APIEntry], library: str, cfg: DictConfig) -> list[APIEntry]:
    """Suffix cap, then the candidate cap with MMR selection, on the union."""
    s1 = cfg.construction.stage1
    kept, capped = apply_suffix_cap(entries, int(s1.suffix_cap.min_group_size),
                                    int(s1.suffix_cap.max_per_suffix))
    if capped:
        logger.info(f"{library}: suffix cap removed {sum(capped.values())} entries")
    cap = max_novel(library, cfg)
    if not cap or len(kept) <= cap:
        return kept
    if not s1.diversity.enabled:
        return top_by_quality(kept, cap)
    return mmr_select(kept, cap, lambda_=float(s1.diversity.lambda_),
                      model_name=str(s1.diversity.model),
                      max_doc_chars=int(s1.diversity.max_doc_chars))


def library_pool(library: str, cfg: DictConfig, maps_dir: Path
                 ) -> tuple[list[APIEntry], dict[str, dict[str, str]], dict, dict[str, str]]:
    """``(candidates, membership, counts, map_sha256)`` of one library."""
    lib = load_library_config(library)
    package, new_v, domain = str(lib.package), str(lib.new_version), str(lib.domain)
    s1 = cfg.construction.stage1
    new_map = load_api_map(maps_dir, package, new_v)
    if new_map is None:
        raise FileNotFoundError(f"no API map for {package}=={new_v} in {maps_dir}")
    shas = {f"{package}=={new_v}": file_sha256(api_map_path(maps_dir, package, new_v))}
    boilerplate = boilerplate_docstrings(new_map, int(s1.boilerplate_min_shared))

    union: dict[str, APIEntry] = {}
    per_model: dict[str, dict[str, str]] = {}
    dropped_boiler = dropped_failed = 0
    for model, old_v in library_old_versions(library, cfg).items():
        if old_v is None:
            kept = entry_filters(list(new_map.values()), library, cfg)
            for e in kept:
                e.old_version, e.domain, e.new_version = None, domain, new_v
                e.extra["no_pre_boundary_release"] = True
            kinds = {e.api_name: "added" for e in kept}
        else:
            old_map = load_api_map(maps_dir, package, old_v)
            if old_map is None:
                raise FileNotFoundError(f"no API map for {package}=={old_v} in {maps_dir}")
            shas[f"{package}=={old_v}"] = file_sha256(api_map_path(maps_dir, package, old_v))
            added, modified = diff_maps(old_map, new_map, old_v, new_v, domain,
                                        bool(s1.include_modified))
            failures = load_map_health(maps_dir, package, old_v)
            if failures:
                n = len(added)
                added = [e for e in added if not under_failed_subtree(e.api_name, failures)]
                dropped_failed += n - len(added)
            kept = entry_filters(added + modified, library, cfg)
            kinds = {e.api_name: ("modified" if e.is_modified else "added") for e in kept}
        n = len(kept)
        kept = [e for e in kept if (e.docstring or "").strip()[:400] not in boilerplate]
        dropped_boiler += n - len(kept)
        names = {e.api_name for e in kept}
        per_model[model] = {k: v for k, v in kinds.items() if k in names}
        for e in kept:
            union.setdefault(e.api_name, e)

    capped = union_set_filters(list(union.values()), library, cfg)
    keep = {e.api_name for e in capped}
    membership = {e.api_name: {m: k[e.api_name] for m, k in per_model.items() if e.api_name in k}
                  for e in capped}
    counts = {"union_before_cap": len(union), "union_after_cap": len(keep),
              "per_model": {m: sum(1 for a in k if a in keep) for m, k in per_model.items()},
              "dropped_boilerplate_docstring": dropped_boiler,
              "dropped_under_failed_subtree": dropped_failed}
    logger.info(f"{library}: union {len(union)} -> {len(keep)} candidates "
                + " ".join(f"{m}={c}" for m, c in counts["per_model"].items()))
    return capped, membership, counts, shas


def _portable(record):
    """Replace absolute site-packages prefixes by ``<site-packages>/``."""
    if isinstance(record, str):
        return _SITE_PACKAGES_RE.sub("<site-packages>/", record)
    if isinstance(record, list):
        return [_portable(x) for x in record]
    if isinstance(record, dict):
        return {k: _portable(v) for k, v in record.items()}
    return record


def freeze_pool(libraries: list[str], cfg: DictConfig, maps_dir: Path, out_dir: Path) -> dict:
    """Write ``candidates.jsonl``, ``membership.csv`` and ``manifest.json``."""
    candidates: dict[str, APIEntry] = {}
    membership: dict[str, dict[str, str]] = {}
    per_library, shas = {}, {}
    for library in libraries:
        cands, memb, counts, sha = library_pool(library, cfg, maps_dir)
        for e in cands:
            candidates[e.api_name] = e
        membership.update(memb)
        per_library[library] = counts
        shas.update(sha)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidates.jsonl").open("w") as f:
        for name in sorted(candidates):
            f.write(json.dumps(_portable(candidates[name].model_dump())) + "\n")
    models = sorted({m for row in membership.values() for m in row})
    with (out_dir / "membership.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["api_name", "library", *models])
        for name in sorted(membership):
            w.writerow([name, candidates[name].library, *[membership[name].get(m, "") for m in models]])
    s1 = cfg.construction.stage1
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_candidates": len(candidates),
        "per_library": per_library,
        "boundaries": boundaries(cfg),
        "caps": {lib: max_novel(lib, cfg) for lib in libraries},
        "stage1": OmegaConf.to_container(
            OmegaConf.masked_copy(s1, ["filters", "suffix_cap", "diversity", "include_modified",
                                       "boilerplate_min_shared"]), resolve=True),
        "api_map_sha256": shas,
    }
    write_json(out_dir / "manifest.json", manifest)
    logger.info(f"frozen pool: {len(candidates)} candidates -> {out_dir}")
    return manifest


def verify_pool(pool_dir: Path, cfg: DictConfig, maps_dir: Path) -> bool:
    """Consistency checks of a frozen pool.

    Artifacts agree; per-boundary sizes shrink with a later boundary; candidate caps hold; and
    every API that is ``added`` for a later boundary but absent from the primary boundary's
    added set was removed and re-introduced (present in the primary ``old_version``, absent from
    the later one), the only way added sets legitimately fail to nest.
    """
    rows = list(csv.DictReader((pool_dir / "membership.csv").open()))
    cands = [json.loads(line) for line in (pool_dir / "candidates.jsonl").open()]
    bounds = boundaries(cfg)
    models = [m for m in bounds if rows and m in rows[0]]
    checks: list[tuple[str, bool, str]] = []
    checks.append(("artifacts agree", {c["api_name"] for c in cands} == {r["api_name"] for r in rows},
                   f"{len(cands)} candidates"))
    tot = {m: sum(1 for r in rows if r[m]) for m in models}
    order = sorted(models, key=lambda m: bounds[m])
    checks.append(("sizes shrink with a later boundary",
                   all(tot[a] >= tot[b] for a, b in zip(order, order[1:])),
                   " ".join(f"{m}={tot[m]}" for m in order)))
    primary = order[0] if order else None
    early_added = {r["api_name"] for r in rows if primary and r[primary] == "added"}
    keys: dict[tuple[str, str], set[str]] = {}

    def map_keys(library: str, version: str | None) -> set[str]:
        if version is None:
            return set()
        if (library, version) not in keys:
            m = load_api_map(maps_dir, str(load_library_config(library).package), version)
            keys[(library, version)] = set(m or {})
        return keys[(library, version)]

    unexplained = []
    for r in rows:
        olds = library_old_versions(r["library"], cfg)
        for m in models:
            if m == primary or r[m] != "added" or r["api_name"] in early_added:
                continue
            if not (r["api_name"] in map_keys(r["library"], olds[primary])
                    and r["api_name"] not in map_keys(r["library"], olds[m])):
                unexplained.append((r["api_name"], m))
    checks.append(("non-nested added APIs are remove-then-re-add", not unexplained,
                   f"{len(unexplained)} unexplained"))
    per_lib: dict[str, int] = {}
    for r in rows:
        per_lib[r["library"]] = per_lib.get(r["library"], 0) + 1
    over = [(lib, n) for lib, n in per_lib.items() if max_novel(lib, cfg) and n > max_novel(lib, cfg)]
    checks.append(("candidate caps hold", not over, str(over)))
    for name, ok, detail in checks:
        logger.info(f"[pool check] {name}: {'PASS' if ok else 'FAIL'} ({detail})")
    return all(ok for _, ok, _ in checks)
