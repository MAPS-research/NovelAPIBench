"""MEMIT (Meng et al., 2023) edits of the RQ3 training tasks (Appendix D.4).

Each training task is one edit request. The edit prompt is the task's S_name prompt with the
subject, the API's S_name, in its ``API: <name>`` line; the target is the reference solution
(:func:`novelapibench.adaptation.data.memit_requests`). The run uses upstream MEMIT
(``third_party/memit``, see its NOTICE) with the hyperparameters of ``configs/adaptation.yaml``:

1. For every request, a target hidden state ``z`` at layer 9 is optimised (25 Adam steps, lr 0.5,
   loss read at layer 27, KL weight 0.0625, norm clamp 0.75) at the last subject token, over the
   edit prompt prefixed by each of MEMIT's context templates. Given the templates, the ``z`` of one
   request does not depend on the others, so they can be computed in shards and are cached as
   ``outputs/adaptation/memit/z_cache/z_layer9_clamp0.75_<task_id>.npz`` (delete the cache after
   changing the model, prompts or hyperparameters).
2. The closed-form update spreads the residuals over ``down_proj`` of layers 5-9, regularised by
   the second-moment statistics of each layer (weight 15,000; :mod:`covariance`).

Upstream samples the context templates from the model without a seed; the templates used for the
paper run are shipped in ``data/adaptation/memit_context_templates.json`` and used by default.
The edited backbone is saved as a full HuggingFace checkpoint and served by vLLM.
"""

from __future__ import annotations

import builtins
import json
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from novelapibench.adaptation.covariance import covariance_path, load_second_moment, missing_layers
from novelapibench.log import logger
from novelapibench.paths import REPO_ROOT, adapters_dir

UPSTREAM_DIR = REPO_ROOT / "third_party" / "memit"
Z_CACHE_TEMPLATE = "z_layer{}_clamp{}_{}.npz"   # upstream cache_template: (layer, clamp, case_id)


def import_upstream() -> Any:
    """Put the vendored MEMIT packages (``memit``, ``rome``, ``util``) on the path; return memit_main."""
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import memit.memit_main as memit_main

    return memit_main


def add_config_aliases(model: Any) -> None:
    """Upstream reads GPT-2 style config names (``n_embd``, ``n_positions``)."""
    cfg = model.config
    if not hasattr(cfg, "n_embd"):
        cfg.n_embd = cfg.hidden_size
    if not hasattr(cfg, "n_positions"):
        cfg.n_positions = getattr(cfg, "max_position_embeddings", 4096)


def _remove_config_aliases(model: Any) -> None:
    for name in ("n_embd", "n_positions"):
        if name in model.config.__dict__:
            delattr(model.config, name)


def hparams(mcfg: DictConfig) -> Any:
    import_upstream()
    from memit.memit_hparams import MEMITHyperParams

    return MEMITHyperParams(**OmegaConf.to_container(mcfg.hparams, resolve=True))


def z_cache_dir() -> Path:
    return adapters_dir() / "memit" / "z_cache"


def _context_templates(mcfg: DictConfig, model: Any, tokenizer: Any) -> list[list[str]]:
    """The shipped templates, or (``context_templates: null``) one sample shared by all shards."""
    memit_main = import_upstream()
    if mcfg.context_templates:
        path = Path(mcfg.context_templates)
        path = path if path.is_absolute() else REPO_ROOT / path
    else:
        path = z_cache_dir() / "context_templates.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            memit_main.CONTEXT_TEMPLATES_CACHE = None
            path.write_text(json.dumps(memit_main.get_context_templates(model, tokenizer)))
    templates = json.loads(path.read_text())
    memit_main.CONTEXT_TEMPLATES_CACHE = templates
    logger.info(f"MEMIT context templates from {path}")
    return templates


@contextmanager
def _quiet_upstream(total: int) -> Iterator[None]:
    """Upstream prints several lines per request (with the full prompt); keep a progress line."""
    original = builtins.print
    noisy = ("Computing right vector", "Lookup index found", "Rewrite layer is", "Tying optimization",
             "loss ", "Init norm", "Cached k/v pair", "Cached context templates",
             "MEMIT request sample", "Retrieving covariance")
    count = 0

    def filtered(*args, **kwargs):
        nonlocal count
        msg = str(args[0]).lstrip() if args else ""
        if msg.startswith("Recording initial value of v*"):
            count += 1
            if count == 1 or count % 50 == 0 or count == total:
                logger.info(f"MEMIT: z vector {count}/{total}")
            return
        if not msg.startswith(noisy):
            original(*args, **kwargs)

    builtins.print = filtered
    try:
        yield
    finally:
        builtins.print = original


def compute_z(model: Any, tokenizer: Any, requests: list[dict], hp: Any, templates: list[list[str]],
              cache_dir: Path) -> int:
    """Optimise and cache the ``z`` of each request that is not cached yet (upstream's z loop)."""
    memit_main = import_upstream()
    z_layer = hp.layers[-1]
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in requests
            if not (cache_dir / Z_CACHE_TEMPLATE.format(z_layer, hp.clamp_norm_factor, r["case_id"])).exists()]
    with _quiet_upstream(len(todo)):
        for request in todo:
            path = cache_dir / Z_CACHE_TEMPLATE.format(z_layer, hp.clamp_norm_factor, request["case_id"])
            request = deepcopy(request)
            if request["target_new"]["str"][0] != " ":   # as execute_memit: leading space
                request["target_new"]["str"] = " " + request["target_new"]["str"]
            z = memit_main.compute_z(model, tokenizer, request, hp, z_layer, templates)
            tmp = path.with_name(path.stem + ".partial.npz")
            np.savez(tmp, v_star=z.detach().cpu().numpy())
            tmp.rename(path)
    return len(todo)


def _load_backbone(model_cfg: DictConfig) -> tuple[Any, Any]:
    from novelapibench.adaptation.sft import load_backbone

    tokenizer, model = load_backbone(model_cfg)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    add_config_aliases(model)
    return tokenizer, model


def train_memit_shard(requests: list[dict], acfg: DictConfig, model_cfg: DictConfig,
                      shard: int, num_shards: int) -> int:
    """Compute the ``z`` vectors of ``requests[shard::num_shards]`` (no weight update)."""
    tokenizer, model = _load_backbone(model_cfg)
    hp = hparams(acfg.memit)
    templates = _context_templates(acfg.memit, model, tokenizer)
    mine = requests[shard::num_shards]
    t0 = time.time()
    n = compute_z(model, tokenizer, mine, hp, templates, z_cache_dir())
    logger.info(f"MEMIT shard {shard}/{num_shards}: {len(mine)} requests, {n} computed "
                f"({(time.time() - t0) / 60:.1f} min)")
    return n


def train_memit(requests: list[dict], acfg: DictConfig, model_cfg: DictConfig, output_dir: Path) -> Path:
    """Edit the backbone with all requests and save the edited checkpoint to ``output_dir``."""
    import gc

    model_short = model_cfg.short_name
    ccfg = acfg.covariance
    if missing := missing_layers(ccfg, model_short):
        raise FileNotFoundError(f"MEMIT: no second-moment statistics for layers {missing} "
                                f"(expected {covariance_path(ccfg, model_short, missing[0])}); "
                                f"run scripts/train_adaptation.py --covariance first")
    tokenizer, model = _load_backbone(model_cfg)
    hp = hparams(acfg.memit)
    memit_main = import_upstream()
    templates = _context_templates(acfg.memit, model, tokenizer)
    # Hand upstream the cached statistics (get_cov keys them by model name and module).
    model_key = model.config._name_or_path.replace("/", "_")
    for layer in hp.layers:
        module = hp.rewrite_module_tmp.format(layer)
        memit_main.COV_CACHE[(model_key, module)] = load_second_moment(
            covariance_path(ccfg, model_short, layer)).float().to("cpu")

    t0 = time.time()
    n = compute_z(model, tokenizer, requests, hp, templates, z_cache_dir())
    logger.info(f"MEMIT: {n} z vectors computed, {len(requests) - n} read from {z_cache_dir()}")
    with _quiet_upstream(0):
        model, _ = memit_main.apply_memit_to_model(
            model, tokenizer, requests, hp, copy=False, return_orig_weights=False,
            cache_template=str(z_cache_dir() / Z_CACHE_TEMPLATE))
    minutes = (time.time() - t0) / 60
    logger.info(f"MEMIT: {len(requests)} edits applied to layers {hp.layers} in {minutes:.1f} min")

    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_config_aliases(model)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    meta = {"method": "memit", "model": model_cfg.name, "revision": model_cfg.get("revision"),
            "n_edits": len(requests), "edit_minutes": round(minutes, 1),
            "subject_template": requests[0]["prompt"].split("{}", 1)[0] if requests else "",
            "hparams": OmegaConf.to_container(acfg.memit.hparams, resolve=True)}
    (output_dir / "edit_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"MEMIT: edited model saved to {output_dir}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir
