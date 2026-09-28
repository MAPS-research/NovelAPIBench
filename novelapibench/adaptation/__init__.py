"""Parametric adaptation (RQ3): SFT, RAFT, GRACE, MEMIT and AlphaEdit-LoRA.

All methods are trained on the RQ3 training split of the primary backbone
(``data/benchmark/splits/rq3_train.txt``) and write one artifact under
``outputs/adaptation/<method>/``, which ``scripts/run_inference.py --experiment rq3`` loads.
Training: ``scripts/train_adaptation.py``; settings: ``configs/adaptation.yaml``.

    data            training examples of every method
    sft             SFT and RAFT (LoRA, TRL SFTTrainer)
    grace           GRACE codebook: training and ``attach_grace_codebook`` for inference
    covariance      second-moment statistics (MEMIT) and the AlphaEdit projector
    memit           MEMIT edits with the vendored upstream code (``third_party/memit``)
    alphaedit_lora  LoRA with AlphaEdit's null-space projection
"""

from __future__ import annotations

from pathlib import Path

from novelapibench.paths import adapters_dir

#: method -> (artifact kind, directory name). Kinds: a LoRA adapter served by vLLM, a full
#: edited checkpoint (MEMIT), or a GRACE codebook (HuggingFace backend).
METHODS: dict[str, tuple[str, str]] = {
    "base": ("none", ""),
    "sft": ("adapter", "sft/adapter"),
    "raft": ("adapter", "raft/adapter"),
    "grace": ("grace", "grace/codebook"),
    "memit": ("edited_model", "memit/edited_model"),
    "alphaedit_lora": ("adapter", "alphaedit_lora/adapter"),
}


def artifact_path(method: str) -> Path | None:
    kind, rel = METHODS[method]
    return None if kind == "none" else adapters_dir() / rel


def llm_kwargs(method: str) -> dict:
    """Keyword arguments of ``LocalLLM`` that load ``method``'s artifact."""
    kind, _ = METHODS[method]
    path = artifact_path(method)
    if kind == "none":
        return {}
    if path is None or not path.exists():
        raise FileNotFoundError(f"{method}: no trained artifact at {path}; "
                                f"run scripts/train_adaptation.py --method {method}")
    return {"adapter": {"adapter_path": path}, "grace": {"grace_path": path},
            "edited_model": {"edited_model_path": path}}[kind]
