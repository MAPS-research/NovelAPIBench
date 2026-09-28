"""Train the RQ3 parametric-adaptation methods on the RQ3 training split (Appendix D.4).

Examples:
    # LoRA fine-tuning: SFT (no-knowledge prompts) and RAFT (prompts with bundles)
    python scripts/train_adaptation.py --method sft
    python scripts/train_adaptation.py --method raft
    # GRACE codebook
    python scripts/train_adaptation.py --method grace
    # Second-moment statistics of layers 5-9 and the AlphaEdit projector (MEMIT, AlphaEdit-LoRA)
    python scripts/train_adaptation.py --covariance
    # MEMIT: optionally compute the z vectors in parallel shards first, then edit
    python scripts/train_adaptation.py --method memit --shard 0 --num-shards 6
    python scripts/train_adaptation.py --method memit
    # AlphaEdit-LoRA
    python scripts/train_adaptation.py --method alphaedit_lora
    # Write the training examples only (no GPU needed)
    python scripts/train_adaptation.py --method raft --data-only
    # Quick check: 8 tasks, 1 epoch, statistics over 1,000 texts, artifacts under outputs/debug/
    python scripts/train_adaptation.py --covariance --debug
    python scripts/train_adaptation.py --method memit --debug

Each method writes its training examples to outputs/adaptation/<method>/train.jsonl and its
artifact to the path in novelapibench.adaptation.METHODS, where
`scripts/run_inference.py --experiment rq3` loads it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novelapibench.adaptation import artifact_path  # noqa: E402
from novelapibench.adaptation.data import build_examples  # noqa: E402
from novelapibench.benchmark import load_bundles, load_split  # noqa: E402
from novelapibench.config import load_config, load_model_config  # noqa: E402
from novelapibench.io import write_jsonl  # noqa: E402
from novelapibench.log import logger, setup_logging  # noqa: E402
from novelapibench.paths import adapters_dir, use_debug_outputs  # noqa: E402

TRAINED = ["sft", "raft", "grace", "memit", "alphaedit_lora"]
#: File whose presence marks a finished artifact.
DONE_MARKER = {"sft": "adapter_config.json", "raft": "adapter_config.json", "grace": "codebook.pt",
               "memit": "config.json", "alphaedit_lora": "adapter_config.json"}
DEBUG_LIMIT = 8
DEBUG_OVERRIDES = ["adaptation.sft.num_epochs=1", "adaptation.raft.num_epochs=1",
                   "adaptation.alphaedit_lora.num_epochs=1", "adaptation.covariance.n_samples=1000"]


def covariance(acfg, model_cfg) -> None:
    import gc

    import torch

    from novelapibench.adaptation.covariance import (compute_covariance, compute_projector,
                                                     missing_layers)
    from novelapibench.adaptation.sft import load_backbone

    if missing_layers(acfg.covariance, model_cfg.short_name):
        tokenizer, model = load_backbone(model_cfg)
        compute_covariance(acfg.covariance, model_cfg, model, tokenizer)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    compute_projector(acfg.covariance, model_cfg.short_name,
                      float(acfg.alphaedit_lora.projection_threshold))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=TRAINED)
    ap.add_argument("--covariance", action="store_true",
                    help="compute the second-moment statistics and the AlphaEdit projector")
    ap.add_argument("--limit", type=int, default=None, help="train on the first N examples only")
    ap.add_argument("--debug", action="store_true",
                    help=f"{DEBUG_LIMIT} examples, 1 epoch, statistics over 1,000 texts; "
                         "outputs under outputs/debug/")
    ap.add_argument("--data-only", action="store_true", help="write the training examples and stop")
    ap.add_argument("--shard", type=int, default=None, help="MEMIT: compute z vectors of one shard")
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if the artifact exists")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. adaptation.sft.num_epochs=1")
    args = ap.parse_args()
    if not args.method and not args.covariance:
        ap.error("pass --method and/or --covariance")
    if (args.shard is None) != (args.num_shards is None) or (args.shard is not None and args.method != "memit"):
        ap.error("--shard and --num-shards go together, with --method memit")
    setup_logging()
    if args.debug:
        use_debug_outputs()

    overrides = (DEBUG_OVERRIDES if args.debug else []) + list(args.overrides)
    limit = args.limit if args.limit is not None else (DEBUG_LIMIT if args.debug else None)
    cfg = load_config(overrides, "adaptation")
    acfg = cfg.adaptation
    model_cfg = load_model_config(acfg.model)

    if args.covariance:
        covariance(acfg, model_cfg)
    if not args.method:
        return

    method = args.method
    tasks = load_split(acfg.split, final=False)
    examples = build_examples(method, acfg, tasks, load_bundles())
    if limit is not None:
        examples = examples[:limit]
    data_path = adapters_dir() / method / "train.jsonl"
    write_jsonl(data_path, examples)
    logger.info(f"{method}: {len(examples)} training examples from {len(tasks)} tasks -> {data_path}")
    if method == "raft":
        dropped = sum(not ex["target_shown"] for ex in examples)
        logger.info(f"raft: target bundle left out of {dropped}/{len(examples)} examples")
    if args.data_only:
        return

    out = artifact_path(method)
    if args.shard is None and (out / DONE_MARKER[method]).exists() and not args.force:
        logger.info(f"{method}: artifact exists at {out}; pass --force to retrain")
        return
    if method in ("sft", "raft"):
        from novelapibench.adaptation.sft import train_lora
        train_lora(method, examples, acfg[method], model_cfg, out)
    elif method == "grace":
        from novelapibench.adaptation.grace import train_grace
        train_grace(examples, acfg.grace, model_cfg, out)
    elif method == "memit":
        from novelapibench.adaptation.memit import train_memit, train_memit_shard
        if args.shard is not None:
            train_memit_shard(examples, acfg, model_cfg, args.shard, args.num_shards)
        else:
            train_memit(examples, acfg, model_cfg, out)
    else:
        from novelapibench.adaptation.alphaedit_lora import train_alphaedit_lora
        train_alphaedit_lora(examples, acfg, model_cfg, out)


if __name__ == "__main__":
    main()
