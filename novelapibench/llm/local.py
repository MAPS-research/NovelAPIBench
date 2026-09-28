"""Local inference for the evaluated backbones (vLLM; HuggingFace for GRACE).

Every prompt is sent as one user message through the backbone's own chat template, with greedy
decoding for pass@1 and a repetition penalty of 1.0 (Appendix D.1). Reasoning traces of
thinking models are stripped from the returned text and kept in ``raw_text``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from novelapibench.evaluation.extraction import strip_thinking
from novelapibench.log import logger


@dataclass
class Generation:
    text: str            # response with any reasoning trace removed
    raw_text: str        # response as generated
    finish_reason: str = "stop"
    num_tokens: int = 0


def strip_duplicate_bos(rendered: str, tokenizer: Any) -> str:
    """Drop a BOS token that the chat template writes when encoding adds one anyway.

    DeepSeek-R1-Distill's template writes ``<｜begin▁of▁sentence｜>`` into the text and its
    tokenizer also prepends BOS, which would give the model two. No-op for Qwen2.5 (no BOS).
    """
    bos = getattr(tokenizer, "bos_token", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if not bos or bos_id is None or not rendered.startswith(bos):
        return rendered
    try:
        adds = tokenizer("x").input_ids[:1] == [bos_id]
    except Exception:  # noqa: BLE001
        return rendered
    return rendered[len(bos):] if adds else rendered


class LocalLLM:
    """One backbone, optionally with a LoRA adapter, an edited checkpoint or a GRACE codebook.

    Args:
        cfg: project config (``inference`` section).
        model_cfg: backbone config (``configs/models/<model>.yaml``).
        adapter_path: LoRA adapter directory (SFT, RAFT, AlphaEdit-LoRA); served by vLLM.
        edited_model_path: full edited checkpoint (MEMIT); loaded by vLLM as the model.
        grace_path: GRACE codebook directory; forces the HuggingFace backend, because GRACE
            needs a forward hook that vLLM cannot host.
    """

    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, adapter_path: str | Path | None = None,
                 edited_model_path: str | Path | None = None, grace_path: str | Path | None = None):
        if sum(x is not None for x in (adapter_path, edited_model_path, grace_path)) > 1:
            raise ValueError("pass at most one of adapter_path, edited_model_path, grace_path")
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.edited_model_path = str(edited_model_path) if edited_model_path else None
        self.grace_path = str(grace_path) if grace_path else None
        self.backend = "hf" if self.grace_path else "vllm"
        self.thinking_mode = bool(model_cfg.get("thinking_mode", False))
        self.chat_template_kwargs = dict(model_cfg.get("chat_template_kwargs", {}) or {})
        self._model: Any = None
        self._tokenizer: Any = None

    # -- loading -------------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_hf()

    def _load_vllm(self) -> None:
        from vllm import LLM

        icfg = self.cfg.inference
        max_model_len = min(int(icfg.max_model_len), int(self.model_cfg.max_seq_len))
        gpu_util = float(icfg.gpu_memory_utilization)
        kwargs: dict[str, Any] = dict(
            tensor_parallel_size=int(self.model_cfg.get("tensor_parallel_size", 1)),
            max_model_len=max_model_len,
            dtype=str(icfg.dtype),
        )
        if self.edited_model_path:
            kwargs["model"] = self.edited_model_path
            # Loading right after another resident model can leave slightly less free memory.
            gpu_util = min(gpu_util, 0.75)
        else:
            kwargs["model"] = self.model_cfg.name
            if self.model_cfg.get("revision"):
                kwargs["revision"] = self.model_cfg.revision
        kwargs["gpu_memory_utilization"] = gpu_util
        if self.model_cfg.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
        if self.adapter_path:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = int(icfg.max_lora_rank)
        logger.info(f"loading {kwargs['model']} with vLLM (max_model_len={max_model_len})")
        self._model = LLM(**kwargs)
        self._tokenizer = self._model.get_tokenizer()

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from novelapibench.adaptation.grace import attach_grace_codebook

        name = self.model_cfg.name
        rev = self.model_cfg.get("revision")
        trust = bool(self.model_cfg.get("trust_remote_code", False))
        self._tokenizer = AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=trust)
        model = AutoModelForCausalLM.from_pretrained(name, revision=rev, device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     trust_remote_code=trust)
        attach_grace_codebook(model, self.grace_path)
        self._model = model

    def close(self) -> None:
        """Release GPU memory (needed before loading another model in the same process)."""
        import gc

        if self._model is not None:
            shutdown = getattr(self._model, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001
                    pass
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- generation ----------------------------------------------------------------------

    def _chat(self, prompt: str) -> str:
        try:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, **self.chat_template_kwargs)
        except TypeError:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        return strip_duplicate_bos(rendered, self._tokenizer)

    def _post(self, raw: str) -> str:
        return strip_thinking(raw) if self.thinking_mode else raw

    def generate(self, prompts: list[str], temperature: float = 0.0, n: int = 1,
                 max_new_tokens: int | None = None) -> list[list[Generation]]:
        """``n`` generations for each prompt."""
        self._load()
        max_tok = int(max_new_tokens or self.model_cfg.max_new_tokens)
        rp = float(self.cfg.inference.repetition_penalty)
        rendered = [self._chat(p) for p in prompts]
        logger.info(f"generating {len(prompts)} prompts x {n} (T={temperature}, max_tokens={max_tok}, "
                    f"repetition_penalty={rp}, backend={self.backend})")
        if self.backend == "vllm":
            return self._generate_vllm(rendered, temperature, n, max_tok, rp)
        return self._generate_hf(rendered, temperature, n, max_tok, rp)

    def _generate_vllm(self, prompts, temperature, n, max_tok, rp):
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tok, n=n, stop=[],
                                repetition_penalty=rp)
        kwargs: dict[str, Any] = {}
        if self.adapter_path:
            from vllm.lora.request import LoRARequest
            kwargs["lora_request"] = LoRARequest("adapter", 1, self.adapter_path)
        outputs = self._model.generate(prompts, params, **kwargs)
        return [[Generation(text=self._post(s.text), raw_text=s.text,
                            finish_reason=s.finish_reason or "stop", num_tokens=len(s.token_ids))
                 for s in out.outputs] for out in outputs]

    def _generate_hf(self, prompts, temperature, n, max_tok, rp):
        import torch
        from tqdm import tqdm

        results = []
        for prompt in tqdm(prompts, desc="generate (HF)"):
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            plen = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=max_tok,
                                           temperature=max(temperature, 1e-6),
                                           do_sample=temperature > 0, num_return_sequences=n,
                                           repetition_penalty=rp)
            gens = []
            for seq in out:
                raw = self._tokenizer.decode(seq[plen:], skip_special_tokens=True)
                gens.append(Generation(text=self._post(raw), raw_text=raw, num_tokens=len(seq) - plen))
            results.append(gens)
        return results
