"""Retrieval setup for RQ2 (Appendix D.3).

For each domain and knowledge condition, every bundle in the domain's retrieval pool is rendered
with the condition's components, split on whitespace and cut into windows of 512 words starting
every 448 words (64-word overlap); each window is rejoined with single spaces, so line breaks and
indentation are not preserved. Chunks are embedded with BAAI/bge-small-en-v1.5 (CLS pooling,
L2-normalized) and searched exactly by inner product (FAISS ``IndexFlatIP``). The query is the task
description followed by the context code, encoded without the model's retrieval instruction.
The encoder truncates chunks and queries to its 512-token limit, so a longer chunk is represented
by its beginning; the generator receives the five highest-scoring chunks untruncated, in rank
order, joined by ``---`` separators, in place of the oracle documentation.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from novelapibench.benchmark import retrieval_pool
from novelapibench.knowledge import Condition, render_knowledge
from novelapibench.log import logger
from novelapibench.paths import retrieval_index_dir
from novelapibench.schemas import Task

CHUNK_OVERLAP_WORDS = 64
SEPARATOR = "\n\n---\n\n"


def chunk_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    step = max(1, chunk_size - CHUNK_OVERLAP_WORDS)
    return [c for c in (" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)) if c]


def _encoder(cfg: DictConfig):
    from sentence_transformers import SentenceTransformer
    r = cfg.retrieval
    return SentenceTransformer(str(r.embedding_model), revision=r.get("embedding_revision"))


def build_index(cfg: DictConfig, domain: str, condition: Condition, index_dir: Path, encoder=None) -> None:
    import faiss
    import numpy as np

    texts, meta = [], []
    for b in retrieval_pool(domain):
        full = render_knowledge(b, condition)
        if not full:
            continue
        for i, chunk in enumerate(chunk_words(full, int(cfg.retrieval.chunk_words))):
            texts.append(chunk)
            meta.append({"api_name": b.api_name, "library": b.library, "chunk_idx": i})
    encoder = encoder or _encoder(cfg)
    emb = np.array(encoder.encode(texts, show_progress_bar=True, batch_size=64), dtype="float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "texts.json").write_text(json.dumps(texts))
    (index_dir / "metadata.json").write_text(json.dumps(meta))
    logger.info(f"indexed {len(texts)} chunks for {domain}/{condition.value} -> {index_dir}")


class Retriever:
    """Top-k retrieval over one domain's index for one knowledge condition."""

    def __init__(self, cfg: DictConfig, domain: str, condition: Condition, encoder=None):
        import faiss

        self.cfg = cfg
        self.top_k = int(cfg.retrieval.top_k)
        index_dir = retrieval_index_dir(domain, condition.value)
        self.encoder = encoder or _encoder(cfg)
        if not (index_dir / "index.faiss").exists():
            build_index(cfg, domain, condition, index_dir, self.encoder)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.texts = json.loads((index_dir / "texts.json").read_text())
        self.metadata = json.loads((index_dir / "metadata.json").read_text())

    @staticmethod
    def query(task: Task) -> str:
        return f"{task.description}\n{task.context_code or ''}"

    def search(self, task: Task, k: int | None = None) -> list[int]:
        import numpy as np
        q = np.array(self.encoder.encode([self.query(task)], normalize_embeddings=True), dtype="float32")
        _, idx = self.index.search(q, min(k or self.top_k, len(self.texts)))
        return [int(i) for i in idx[0] if 0 <= i < len(self.texts)]

    def knowledge_text(self, task: Task) -> str:
        return SEPARATOR.join(self.texts[i] for i in self.search(task))

    def target_rank(self, task: Task, k: int = 5) -> int | None:
        """1-based rank of the first chunk from the task's target API within the top ``k``."""
        for rank, i in enumerate(self.search(task, k), 1):
            if self.metadata[i]["api_name"] == task.api_name:
                return rank
        return None
