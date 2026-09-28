"""Stage 2 driver: one knowledge bundle per candidate API (Appendix B.2, "Stage 2").

For each candidate: S (``surface.extract_s``), E (``surface.generate_examples``), C
(``implementation``, computed once and shared with the grounding choice), and M
(``mechanism``). A bundle is discarded when the API is primarily file I/O, when no example
is executed or static, when C is missing or longer than 12,000 characters, or when a
component the knowledge conditions rely on is empty (no parameters and no return type, no
usable example, no mechanism text, no implementation).

Output: ``outputs/construction/stage2/<library>.jsonl`` (``KnowledgeBundle`` records, appended
as they complete; a re-run skips APIs already written) and ``<library>.summary.json``.
"""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm

from novelapibench.config import load_library_config
from novelapibench.construction.envs import stage_python
from novelapibench.construction.schemas import APIEntry
from novelapibench.construction.stage2.implementation import extract_implementation
from novelapibench.construction.stage2.mechanism import (
    changelog_notes_for, extract_mechanism, fetch_changelog)
from novelapibench.construction.stage2.surface import extract_s, generate_examples
from novelapibench.io import iter_jsonl, write_json
from novelapibench.llm.strong import StrongLLM
from novelapibench.log import logger
from novelapibench.schemas import KnowledgeBundle


def build_bundle(entry: APIEntry, cfg: DictConfig, llm: StrongLLM, env_python: str,
                 changelog: str | None) -> KnowledgeBundle:
    lib = load_library_config(entry.library)
    lc = lib.get("construction", {})
    version = str(lib.new_version)
    surface = extract_s(entry, version, llm)
    surface.examples = generate_examples(entry, version, surface, llm, cfg, env_python)

    timeout = int(lc.get("implementation_timeout_seconds",
                         cfg.construction.stage2.implementation.timeout_seconds))
    memo: dict[str, str | None] = {}

    def implementation() -> str | None:
        if "code" not in memo:
            memo["code"] = extract_implementation(entry.api_name, env_python, timeout)
        return memo["code"]

    mechanism = extract_mechanism(entry, str(lc.get("doc_url", "")), cfg, llm, implementation,
                                  changelog_notes_for(changelog, entry.api_name))
    return KnowledgeBundle(
        api_name=entry.api_name, library=entry.library, domain=entry.domain,
        s_name=entry.api_name, s_param=surface.s_param, return_type=surface.return_type,
        examples=surface.examples, mechanism=mechanism, implementation=implementation(),
        is_modified=entry.is_modified, old_signature=entry.extra.get("old_signature"),
        old_parameters=entry.extra.get("old_parameters"),
        requires_file_io=surface.requires_file_io)


def drop_reason(bundle: KnowledgeBundle, cfg: DictConfig) -> str | None:
    """Why a bundle is discarded (None: kept). May blank an oversized implementation."""
    s2 = cfg.construction.stage2
    if s2.exclude_file_io and bundle.requires_file_io:
        return "file_io"
    if not any(ex.status in ("executed", "static") for ex in bundle.examples):
        return "no_valid_example"
    max_chars = int(s2.implementation.max_chars)
    if max_chars and bundle.implementation and len(bundle.implementation) > max_chars:
        bundle.implementation = None
    if not (bundle.implementation or "").strip():
        return "no_implementation"
    # Completeness: every knowledge condition must have its component.
    if not bundle.s_param and not bundle.return_type:
        return "empty_s_param"
    if not bundle.mechanism.text.strip():
        return "empty_mechanism"
    return None


def extract_library(library: str, entries: list[APIEntry], cfg: DictConfig, llm: StrongLLM,
                    out_path: Path) -> list[KnowledgeBundle]:
    lib = load_library_config(library)
    env_python = stage_python(library)
    logger.info(f"Stage 2 {library}: {len(entries)} candidates, execution env {env_python}")
    done = {r["api_name"] for r in iter_jsonl(out_path)} if out_path.exists() else set()
    todo = [e for e in entries if e.api_name not in done]
    url = str(lib.get("construction", {}).get("changelog_url", "") or "")
    changelog = fetch_changelog(url) if url and todo else None

    reasons: Counter = Counter()
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def work(entry: APIEntry) -> tuple[APIEntry, KnowledgeBundle | None, str | None]:
        try:
            return entry, build_bundle(entry, cfg, llm, env_python, changelog), None
        except Exception as exc:  # noqa: BLE001  (one API must not stop the library)
            return entry, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max(1, int(cfg.construction.stage2.workers))) as pool, \
            out_path.open("a") as fh:
        futures = [pool.submit(work, e) for e in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Stage 2 [{library}]"):
            entry, bundle, err = fut.result()
            if err is not None:
                logger.warning(f"{entry.api_name}: extraction failed ({err})")
                reasons["extraction_error"] += 1
                continue
            reason = drop_reason(bundle, cfg)
            if reason:
                reasons[reason] += 1
                continue
            with lock:
                fh.write(bundle.model_dump_json() + "\n")
                fh.flush()
            reasons["kept"] += 1

    bundles = [KnowledgeBundle(**r) for r in iter_jsonl(out_path)]
    summary = {"library": library, "candidates": len(entries), "resumed": len(done),
               "bundles": len(bundles), "this_run": dict(reasons)}
    write_json(out_path.with_suffix(".summary.json"), summary)
    logger.info(f"Stage 2 {library}: {len(bundles)} bundles ({dict(reasons)})")
    return bundles
