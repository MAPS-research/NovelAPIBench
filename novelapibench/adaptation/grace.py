"""GRACE (Hartvigsen et al., 2023): edits stored in a key-value memory at one layer (Appendix D.4).

Independent implementation of the method as configured in the paper. A :class:`KeyValueMemory`
wraps the ``nn.Linear`` at layer 24's ``up_proj`` of the frozen backbone:

* **Key.** The layer input at one token: the last prompt token while an edit is learned, the last
  token of the current input at inference.
* **Lookup.** If the nearest stored key lies within its radius, that key's value replaces the layer
  output at every position before the key token; otherwise the layer is unchanged.
* **Writing an edit.** Each RQ3 training task is one edit (chat-templated no-knowledge prompt ->
  reference solution), applied sequentially in split order. On the edit's first step, the memory
  is updated with the new key:
  - farther than ``radius0`` plus the nearest entry's radius from every entry: a new entry with
    radius ``radius0``;
  - inside that margin but belonging to a different edit: a new entry, and both radii set to half
    the distance (the old one slightly less);
  - belonging to the same edit but outside its radius: the radius grows to cover it.
  A new entry's value starts random (uniform) and is optimised with Adam (lr 1.0) for up to 100
  steps on the loss over the target tokens, stopping once the loss falls below 0.01.

A candidate value is drawn on every forward pass through a non-empty memory, so the random
stream, and with it the stored values, matches the runs behind the paper. The artifact is
``codebook.pt`` plus ``meta.json``; :func:`attach_grace_codebook` restores it on a freshly loaded
backbone. The memory needs a forward hook, so GRACE is evaluated with HuggingFace generation
instead of vLLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from novelapibench.log import logger


class KeyValueMemory(nn.Module):
    """An ``nn.Linear`` whose output is overridden for inputs that match a stored key."""

    def __init__(self, base: nn.Linear, radius0: float = 1.0):
        super().__init__()
        self.base = base
        self.radius0 = float(radius0)
        self.keys: torch.Tensor | None = None       # (n, in_features), backbone dtype
        self.radii: torch.Tensor | None = None      # (n,), float32
        self.values: nn.Parameter | None = None     # (n, out_features), float32
        self.edit_ids: list[float] = []             # which edit wrote each entry
        self._edit: dict | None = None              # set while an edit is being learned

    @property
    def size(self) -> int:
        return 0 if self.keys is None else self.keys.shape[0]

    # -- editing --------------------------------------------------------------------------

    def start_edit(self, key_position: int, edit_id: float) -> None:
        self._edit = {"position": key_position, "id": edit_id, "first_step": True}

    def stop_edit(self) -> None:
        self._edit = None

    def _append(self, key: torch.Tensor, value: torch.Tensor, radius: float) -> None:
        key = key.detach()
        value = value.detach().to(torch.float32)
        r = torch.tensor([radius], dtype=torch.float32, device=key.device)
        if self.keys is None:
            self.keys, self.radii, new_values = key, r, value
        else:
            self.keys = torch.cat([self.keys, key])
            self.radii = torch.cat([self.radii, r])
            new_values = torch.cat([self.values.detach(), value])
        self.values = nn.Parameter(new_values)
        self.edit_ids.append(self._edit["id"])

    def _write(self, key: torch.Tensor, candidate: torch.Tensor) -> None:
        """Insert ``key`` (one row) into the memory, or widen the entry that covers it."""
        if self.keys is None:
            self._append(key, candidate, self.radius0)
            return
        dist, idx = self._nearest(key)
        d, i = float(dist[0]), int(idx[0])
        if d > self.radius0 + float(self.radii[i]):
            self._append(key, candidate, self.radius0)
        elif self._edit["id"] != self.edit_ids[i]:
            self._append(key, candidate, self.radius0)
            self.radii[i] = dist[0] / 2 - 1e-5
            self.radii[-1] = dist[0] / 2
        elif d > float(self.radii[i]):
            self.radii[i] = dist[0]

    # -- lookup ---------------------------------------------------------------------------

    def _nearest(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Distance to, and index of, the nearest key for each query (float32 distances)."""
        d = torch.cdist(self.keys.float(), queries.float())      # (n, batch)
        return d.min(dim=0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.base(x, *args, **kwargs)
        editing = self._edit is not None
        if not editing and self.keys is None:
            return out
        seq_len = x.shape[1]
        pos = min(self._edit["position"], seq_len - 1) if editing else seq_len - 1
        query = x[:, pos, :]
        candidate = torch.rand(1, out.shape[-1], device=x.device)
        if editing and (self.keys is None or self._edit["first_step"]):
            self._write(query, candidate)
            self._edit["first_step"] = False
        dist, idx = self._nearest(query)
        if pos > 0:
            hit = (dist <= self.radii[idx]).view(-1, 1, 1)
            replacement = self.values[idx].unsqueeze(1).expand(-1, pos, -1)
            out[:, :pos] = torch.where(hit, replacement, out[:, :pos])
        return out

    # -- persistence ----------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {"keys": self.keys.detach().cpu(), "values": self.values.detach().cpu(),
                "radii": self.radii.detach().cpu(), "radius0": self.radius0}

    def restore(self, state: dict[str, Any]) -> None:
        device = self.base.weight.device
        self.keys = state["keys"].to(device)
        self.radii = state["radii"].to(device)
        self.values = nn.Parameter(state["values"].to(device), requires_grad=False)
        self.radius0 = float(state["radius0"])


def _submodule(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    """Parent module and attribute name of a dotted path such as ``model.layers.24.mlp.up_proj``."""
    *parents, leaf = path.removesuffix(".weight").split(".")
    module: Any = root
    for p in parents:
        module = module[int(p)] if p.isdigit() else getattr(module, p)
    return module, leaf


def attach_memory(model: nn.Module, layer: str, radius0: float = 1.0) -> KeyValueMemory:
    """Wrap the ``nn.Linear`` at ``layer`` in an (empty) memory and return the wrapper."""
    parent, leaf = _submodule(model, layer)
    base = getattr(parent, leaf)
    if not isinstance(base, nn.Linear):
        raise TypeError(f"GRACE wraps an nn.Linear; {layer} is {type(base).__name__}")
    memory = KeyValueMemory(base, radius0)
    setattr(parent, leaf, memory)
    return memory


def attach_grace_codebook(model: nn.Module, path: str | Path) -> KeyValueMemory:
    """Attach a trained memory (directory with ``codebook.pt`` and ``meta.json``) for inference."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    meta = json.loads((path / "meta.json").read_text())
    memory = attach_memory(model, meta["layer"], meta["radius0"])
    memory.restore(torch.load(path / "codebook.pt", map_location="cpu"))
    logger.info(f"GRACE memory with {memory.size} entries attached at {meta['layer']}")
    return memory


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------


def _encode_edit(tokenizer: Any, prompt: str, target: str, device: torch.device,
                 max_seq_len: int) -> tuple[dict[str, torch.Tensor], int]:
    """Chat-templated prompt + target with the prompt masked from the loss, and the number of
    prompt tokens. Over-long edits lose prompt tokens from the left."""
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    ids = prompt_ids + target_ids
    if len(ids) > max_seq_len:
        room = max_seq_len - len(target_ids)
        if room <= 0:
            ids, n_prompt = ids[-max_seq_len:], 0
        else:
            ids, n_prompt = prompt_ids[-room:] + target_ids, room
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    labels = input_ids.clone()
    labels[0, :n_prompt] = -100
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
            "labels": labels}, n_prompt


def learn_edit(memory: KeyValueMemory, model: nn.Module, batch: dict[str, torch.Tensor],
               n_prompt: int, gcfg: DictConfig) -> None:
    """Write one edit: key at the last prompt token, value fitted to the target tokens."""
    edit_id = float(batch["labels"].float().mean())      # identifies the edit's target
    memory.start_edit(max(n_prompt - 1, 0), edit_id)
    optimizer = None
    for _ in range(int(gcfg.n_iter)):
        loss = model(**batch).loss
        if optimizer is None:  # the entry's value exists only after the first forward pass
            optimizer = torch.optim.Adam([memory.values], lr=float(gcfg.edit_lr))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if float(loss.detach()) < float(gcfg.early_stop_loss):
            break
    memory.stop_edit()


def train_grace(examples: list[dict], gcfg: DictConfig, model_cfg: DictConfig, output_dir: Path) -> Path:
    """Apply one edit per example in order and save the memory to ``output_dir``."""
    import gc

    from novelapibench.adaptation.sft import load_backbone

    if str(gcfg.optimizer).lower() != "adam":
        raise ValueError("GRACE edits are optimised with Adam")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_backbone(model_cfg)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    memory = attach_memory(model, gcfg.layer, float(gcfg.eps))
    device = memory.base.weight.device

    t0 = time.time()
    for i, ex in enumerate(examples):
        batch, n_prompt = _encode_edit(tokenizer, ex["prompt"], ex["target"], device,
                                       int(gcfg.max_seq_len))
        learn_edit(memory, model, batch, n_prompt, gcfg)
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(examples):
            logger.info(f"GRACE: {i + 1}/{len(examples)} edits, {memory.size} entries, "
                        f"{(time.time() - t0) / 60:.1f} min")

    torch.save(memory.state(), output_dir / "codebook.pt")
    meta = {"model": model_cfg.name, "revision": model_cfg.get("revision"), "layer": gcfg.layer,
            "radius0": float(gcfg.eps), "optimizer": "adam", "edit_lr": float(gcfg.edit_lr),
            "n_iter": int(gcfg.n_iter), "early_stop_loss": float(gcfg.early_stop_loss),
            "n_edits": len(examples), "n_entries": memory.size}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"GRACE: memory ({memory.size} entries) saved to {output_dir}")

    del model, memory
    gc.collect()
    torch.cuda.empty_cache()
    return output_dir
