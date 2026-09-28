"""Second-moment statistics of the edited layers (Appendix D.4, "Adaptation methods").

MEMIT regularises its closed-form update with ``C = E[k k^T]``, the uncentred second moment of the
inputs ``k`` to ``down_proj`` in layers 5-9, estimated over 20,000 Wikitext-103 training texts in
FP32 with upstream MEMIT's ``layer_stats`` (a fixed random subset, seed 1). AlphaEdit-LoRA reuses
the same statistics: its projector ``P = U_0 U_0^T`` spans the singular vectors of ``C`` whose
singular values are below the threshold (2e-2), i.e. input directions that the corpus barely uses.

Files, under ``outputs/adaptation/stats/<model>/``::

    <dataset>_stats/<module>_<dtype>_mom2_<n>.npz    upstream cache format (sum and count)
    alphaedit_projector_<dtype>_<n>_<threshold>.pt   [n_layers, d, d] float32 projector
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from novelapibench.log import logger
from novelapibench.paths import adapters_dir


def stats_dir(model: str) -> Path:
    return adapters_dir() / "stats" / model


def covariance_path(ccfg: DictConfig, model: str, layer: int) -> Path:
    module = str(ccfg.module).format(layer)
    return (stats_dir(model) / f"{ccfg.dataset}_stats"
            / f"{module}_{ccfg.dtype}_mom2_{int(ccfg.n_samples)}.npz")


def projector_path(ccfg: DictConfig, model: str, threshold: float) -> Path:
    return stats_dir(model) / f"alphaedit_projector_{ccfg.dtype}_{int(ccfg.n_samples)}_{threshold:g}.pt"


def missing_layers(ccfg: DictConfig, model: str) -> list[int]:
    return [layer for layer in ccfg.layers if not covariance_path(ccfg, model, layer).exists()]


def load_second_moment(path: Path) -> torch.Tensor:
    """``E[k k^T]`` (float32, CPU) from a cached statistic, as upstream ``SecondMoment.moment()``."""
    with np.load(path) as d:
        return torch.from_numpy(d["mom2.mom2"]) / int(d["mom2.count"])


def compute_covariance(ccfg: DictConfig, model_cfg: DictConfig, model: Any, tokenizer: Any) -> None:
    """Estimate and cache the statistics of every configured layer that is not cached yet."""
    from novelapibench.adaptation.memit import add_config_aliases, import_upstream

    import_upstream()
    from rome.layer_stats import layer_stats

    add_config_aliases(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for layer in missing_layers(ccfg, model_cfg.short_name):
        module = str(ccfg.module).format(layer)
        logger.info(f"second moment of {module} over {ccfg.n_samples} {ccfg.dataset} texts")
        # Upstream stores the statistic at <stats_dir>/<model_name>/<dataset>_stats/...
        layer_stats(model, tokenizer, module, adapters_dir() / "stats", str(ccfg.dataset),
                    ["mom2"], model_name=model_cfg.short_name, sample_size=int(ccfg.n_samples),
                    precision=str(ccfg.dtype), download=False)
        logger.info(f"saved {covariance_path(ccfg, model_cfg.short_name, layer)}")


def compute_projector(ccfg: DictConfig, model: str, threshold: float) -> Path:
    """AlphaEdit's null-space projectors of all configured layers, stacked and cached."""
    out = projector_path(ccfg, model, threshold)
    if out.exists():
        return out
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projectors = []
    for layer in ccfg.layers:
        cov = load_second_moment(covariance_path(ccfg, model, layer)).to(torch.float32).to(device)
        u, s, _ = torch.linalg.svd(cov, full_matrices=False)
        keep = s < threshold
        logger.info(f"layer {layer}: {int(keep.sum())}/{s.numel()} directions below {threshold:g}")
        u0 = u[:, keep].contiguous()
        projectors.append((u0 @ u0.T).cpu())
        del cov, u, s, u0
    torch.save(torch.stack(projectors).contiguous(), out)
    logger.info(f"saved AlphaEdit projector {out}")
    return out
