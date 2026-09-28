"""SFT and RAFT: LoRA fine-tuning of the backbone (Section 4.4 RQ3, Appendix D.4).

Each training example (:mod:`novelapibench.adaptation.data`) is a user/assistant pair rendered
with the backbone's chat template into one text. TRL's ``SFTTrainer`` trains a LoRA adapter on
these texts with the language-modelling loss over the whole sequence, prompt included (TRL appends
the EOS token to each text). SFT sees no-knowledge prompts; RAFT sees prompts with the target
bundle among distractor bundles. The adapter is served by vLLM at evaluation.

The LoRA matrices are initialised from PyTorch's default generator before the trainer is built;
the training seed fixes the data order and dropout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from novelapibench.adaptation.data import chat_text
from novelapibench.log import logger


def load_backbone(model_cfg: DictConfig) -> tuple[Any, Any]:
    """Tokenizer and BF16 model of the backbone, placed on the available GPUs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = dict(revision=model_cfg.get("revision"),
                  trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, dtype=torch.bfloat16,
                                                 device_map="auto", **kwargs)
    return tokenizer, model


def lora_config(tcfg: DictConfig) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=int(tcfg.lora_rank),
                      lora_alpha=int(tcfg.lora_alpha), lora_dropout=float(tcfg.lora_dropout),
                      target_modules=list(tcfg.target_modules), bias="none")


def trainer_args(tcfg: DictConfig, output_dir: Path) -> Any:
    """TRL ``SFTConfig`` for one LoRA run (AdamW, the Trainer's default optimizer)."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(tcfg.num_epochs),
        per_device_train_batch_size=int(tcfg.per_device_batch_size),
        gradient_accumulation_steps=int(tcfg.gradient_accumulation_steps),
        learning_rate=float(tcfg.learning_rate),
        warmup_ratio=float(tcfg.warmup_ratio),
        lr_scheduler_type=str(tcfg.lr_scheduler),
        bf16=bool(tcfg.bf16),
        gradient_checkpointing=bool(tcfg.gradient_checkpointing),
        max_length=int(tcfg.max_seq_len),
        dataset_text_field="text",
        seed=int(tcfg.seed),
        data_seed=int(tcfg.seed),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )


def text_dataset(tokenizer: Any, examples: list[dict]) -> Any:
    from datasets import Dataset

    return Dataset.from_dict({"text": [chat_text(tokenizer, ex["messages"]) for ex in examples]})


def write_meta(output_dir: Path, method: str, model_cfg: DictConfig, tcfg: DictConfig,
               n_examples: int, **extra: Any) -> None:
    meta = {"method": method, "model": model_cfg.name, "revision": model_cfg.get("revision"),
            "n_examples": n_examples, "config": OmegaConf.to_container(tcfg, resolve=True), **extra}
    (output_dir / "adaptation_meta.json").write_text(json.dumps(meta, indent=2))


def train_lora(method: str, examples: list[dict], tcfg: DictConfig, model_cfg: DictConfig,
               output_dir: Path) -> Path:
    """Train a LoRA adapter on chat examples (SFT or RAFT) and save it to ``output_dir``."""
    import gc

    import torch
    from peft import get_peft_model
    from trl import SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_backbone(model_cfg)
    tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(model, lora_config(tcfg))
    model.print_trainable_parameters()
    dataset = text_dataset(tokenizer, examples)
    logger.info(f"{method}: {len(examples)} examples, max_seq_len={tcfg.max_seq_len}, "
                f"batch {tcfg.per_device_batch_size} x accumulation {tcfg.gradient_accumulation_steps}")

    trainer = SFTTrainer(model=model, args=trainer_args(tcfg, output_dir), train_dataset=dataset,
                         processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_meta(output_dir, method, model_cfg, tcfg, len(examples))
    logger.info(f"{method}: adapter saved to {output_dir}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir
