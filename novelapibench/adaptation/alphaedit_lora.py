"""AlphaEdit-LoRA: LoRA fine-tuning with AlphaEdit's null-space projection (Appendix D.4).

AlphaEdit (Fang et al., 2025) projects MEMIT's update onto the null space of the preserved-
knowledge keys. This variant transfers the projection to gradient-based LoRA training:

* LoRA adapters (rank 64, alpha 32, dropout 0.1) are attached to ``down_proj`` of every layer and
  only those of layers 5-9, the MEMIT layers, are trained;
* before training and after every optimizer step, the input matrix ``A`` of each trained adapter
  is replaced by ``A P``, where ``P`` projects onto the directions of the layer's input second
  moment with singular value below 2e-2
  (:func:`novelapibench.adaptation.covariance.compute_projector`), so the update ``B A`` stays in
  that subspace;
* data, chat formatting and loss are those of SFT (no-knowledge prompt, loss over the full
  sequence), trained with TRL's SFTTrainer.

The artifact is a PEFT LoRA adapter served by vLLM like the SFT adapters.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from novelapibench.adaptation.covariance import projector_path
from novelapibench.adaptation.sft import (load_backbone, lora_config, text_dataset, trainer_args,
                                          write_meta)
from novelapibench.log import logger

_LAYER = re.compile(r"\.layers\.(\d+)\.")


def freeze_outside(model: Any, layers: set[int], module: str = "down_proj") -> int:
    """Freeze every LoRA parameter outside ``module`` of ``layers``; return the trainable count."""
    n = 0
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        m = _LAYER.search(name)
        if f".{module}." in name and m and int(m.group(1)) in layers:
            n += 1
        else:
            param.requires_grad = False
    return n


def project_lora_a(model: Any, projectors: dict[int, torch.Tensor]) -> None:
    """``A <- A P`` for every trainable ``lora_A`` whose layer has a projector."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" not in name or not param.requires_grad:
                continue
            m = _LAYER.search(name)
            P = projectors.get(int(m.group(1))) if m else None
            if P is None or param.dim() != 2 or P.shape != (param.shape[1], param.shape[1]):
                continue
            param.data.copy_(param.data @ P.to(device=param.device, dtype=param.dtype))


def load_projectors(path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    P = torch.load(path, map_location="cpu")
    if P.dim() != 3 or P.shape[0] != len(layers):
        raise ValueError(f"{path}: expected [{len(layers)}, d, d], got {tuple(P.shape)}")
    return {layer: P[i].contiguous() for i, layer in enumerate(layers)}


def train_alphaedit_lora(examples: list[dict], acfg: DictConfig, model_cfg: DictConfig,
                         output_dir: Path) -> Path:
    """Train the AlphaEdit-LoRA adapter on chat examples and save it to ``output_dir``."""
    import gc

    from peft import get_peft_model
    from transformers import TrainerCallback
    from trl import SFTTrainer

    tcfg = acfg.alphaedit_lora
    layers = [int(x) for x in tcfg.layers]
    p_path = projector_path(acfg.covariance, model_cfg.short_name, float(tcfg.projection_threshold))
    if not p_path.exists():
        raise FileNotFoundError(f"AlphaEdit projector {p_path} is missing; "
                                f"run scripts/train_adaptation.py --covariance first")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_backbone(model_cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = text_dataset(tokenizer, examples)
    model = get_peft_model(model, lora_config(tcfg))
    n_trainable = freeze_outside(model, set(layers))
    if n_trainable == 0:
        raise RuntimeError("AlphaEdit-LoRA: no trainable LoRA parameters in the configured layers")
    model.print_trainable_parameters()

    device = next(p for p in model.parameters() if p.requires_grad).device
    projectors = {layer: P.to(device) for layer, P in load_projectors(p_path, layers).items()}
    project_lora_a(model, projectors)

    class ProjectAfterStep(TrainerCallback):
        """Trainer calls this after every optimizer step (before the gradients are cleared)."""

        def on_optimizer_step(self, args, state, control, **kwargs):
            project_lora_a(model, projectors)

    logger.info(f"AlphaEdit-LoRA: {len(examples)} examples, layers {layers}, projector {p_path}")
    trainer = SFTTrainer(model=model, args=trainer_args(tcfg, output_dir), train_dataset=dataset,
                         processing_class=tokenizer, callbacks=[ProjectAfterStep()])
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_meta(output_dir, "alphaedit_lora", model_cfg, tcfg, len(examples), projector=str(p_path))
    logger.info(f"AlphaEdit-LoRA: adapter saved to {output_dir}")

    del trainer, model, projectors
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir
