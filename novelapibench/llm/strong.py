"""The "strong" model (GPT-5-mini through the OpenAI API).

Used for knowledge extraction, task and scenario generation, the C3 solver, and the failure
classifier. Deterministic calls (temperature 0) and web searches are cached in SQLite, so a
re-run only pays for new prompts. Requires ``OPENAI_API_KEY``.

Note: GPT-5 models accept only their default temperature, so ``temperature`` is not sent to
them; it still distinguishes cache entries and decides what is cached.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from novelapibench.llm.cache import LLMCache, make_cache_key
from novelapibench.log import logger
from novelapibench.paths import REPO_ROOT, cache_dir

_api_retry = retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=lambda rs: logger.warning(
        f"API call failed (attempt {rs.attempt_number}/5): {rs.outcome.exception()}"),
    reraise=True,
)

_USAGE_LOCK = threading.Lock()


def _log_usage(model: str, response: Any) -> None:
    """Append token usage of each paid call to ``$NOVELAPIBENCH_OPENAI_USAGE_LOG`` if set."""
    path = os.environ.get("NOVELAPIBENCH_OPENAI_USAGE_LOG")
    u = getattr(response, "usage", None)
    if not path or u is None:
        return
    det = getattr(u, "completion_tokens_details", None) or getattr(u, "output_tokens_details", None)
    # Chat Completions report prompt/completion tokens, the Responses API input/output tokens.
    rec = {"model": model,
           "prompt_tokens": getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None),
           "completion_tokens": getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None),
           "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det is not None else None}
    with _USAGE_LOCK, open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


class StrongLLM:
    def __init__(self, cfg: DictConfig, cache_path: str | Path | None = None):
        s = cfg.strong_model
        self.model_name = str(s.name)
        self.search_model = str(s.get("search_model", s.name))
        self.reasoning_effort = s.get("reasoning_effort")
        self.max_tokens = int(s.max_tokens)
        self.temperature = float(s.temperature)
        self.timeout = float(s.get("request_timeout_seconds", 120))
        c = cfg.get("cache", {})
        path = Path(cache_path or c.get("path") or cache_dir() / "llm_cache.sqlite")
        if not path.is_absolute():
            path = REPO_ROOT / path
        self._cache = LLMCache(path, enabled=bool(c.get("enabled", True)),
                               journal_mode=c.get("journal_mode"))
        self._client = None

    # -- helpers -------------------------------------------------------------------------

    def _openai(self):
        if self._client is None:
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    def _cached(self, *, call_type: str, model: str, system: str | None, prompt: str,
                temperature: float, max_tokens: int, fn: Callable[[], str]) -> str:
        cacheable = self._cache.enabled and (call_type == "search" or temperature <= 0.0)
        key = None
        if cacheable:
            key = make_cache_key(backend="openai", model=model, call_type=call_type, system=system,
                                 prompt=prompt, temperature=temperature, max_tokens=max_tokens)
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.bump_hit(key)
                return hit
        resp = fn()
        if cacheable and resp:
            self._cache.put(key, backend="openai", model=model, call_type=call_type,
                            temperature=temperature, max_tokens=max_tokens, prompt=prompt,
                            system=system, response=resp)
        return resp

    # -- calls ---------------------------------------------------------------------------

    def generate(self, prompt: str, system: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        temp = self.temperature if temperature is None else float(temperature)
        max_tok = self.max_tokens if max_tokens is None else int(max_tokens)

        @_api_retry
        def call() -> str:
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": prompt}]
            kwargs: dict[str, Any] = {"model": self.model_name, "messages": messages,
                                      "timeout": self.timeout}
            if self.model_name.startswith("gpt-5"):
                # Reasoning tokens count against the completion budget: leave room for them.
                kwargs["max_completion_tokens"] = max_tok + 4096
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["max_completion_tokens"] = max_tok
                kwargs["temperature"] = temp
            response = self._openai().chat.completions.create(**kwargs)
            _log_usage(self.model_name, response)
            return response.choices[0].message.content or ""

        return self._cached(call_type="generate", model=self.model_name, system=system,
                            prompt=prompt, temperature=temp, max_tokens=max_tok, fn=call)

    def generate_with_web_search(self, prompt: str, system: str | None = None,
                                 max_tokens: int | None = None) -> str:
        """Generation with the Responses API ``web_search`` tool (used by knowledge extraction)."""
        max_tok = self.max_tokens if max_tokens is None else int(max_tokens)
        budget = max(max_tok, 4096)  # tool calls count against max_output_tokens

        @_api_retry
        def call() -> str:
            kwargs: dict[str, Any] = {"model": self.search_model, "input": prompt,
                                      "tools": [{"type": "web_search"}],
                                      "max_output_tokens": budget}
            if system:
                kwargs["instructions"] = system
            if self.search_model.startswith("gpt-5") and self.reasoning_effort:
                effort = "low" if self.reasoning_effort == "minimal" else self.reasoning_effort
                kwargs["reasoning"] = {"effort": effort}
            response = self._openai().responses.create(**kwargs)
            _log_usage(self.search_model, response)
            return response.output_text or ""

        return self._cached(call_type="search", model=self.search_model, system=system,
                            prompt=prompt, temperature=0.0, max_tokens=max_tok, fn=call)
