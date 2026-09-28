"""Configuration loading (OmegaConf).

``configs/default.yaml`` holds every setting used for the paper. Backbones live in
``configs/models/<model>.yaml`` and libraries in ``configs/libraries/<library>.yaml``;
benchmark construction and parametric adaptation keep their settings in
``configs/construction.yaml`` and ``configs/adaptation.yaml``.

Any value can be overridden on the command line with OmegaConf dot-list syntax,
e.g. ``inference.tensor_parallel_size=2``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from novelapibench.paths import CONFIG_DIR


def load_config(overrides: list[str] | None = None, *sections: str) -> DictConfig:
    """``configs/default.yaml`` merged with the named section files and dot-list overrides.

    ``sections`` names extra files under ``configs/`` (``"construction"``, ``"adaptation"``);
    each is merged under a top-level key of the same name.
    """
    cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")
    for name in sections:
        cfg = OmegaConf.merge(cfg, {name: OmegaConf.load(CONFIG_DIR / f"{name}.yaml")})
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def load_model_config(model: str) -> DictConfig:
    """Backbone settings from ``configs/models/<model>.yaml`` (``short_name`` added)."""
    path = CONFIG_DIR / "models" / f"{model}.yaml"
    if not path.exists():
        known = ", ".join(list_models())
        raise FileNotFoundError(f"unknown model {model!r}; known: {known}")
    cfg = OmegaConf.load(path)
    cfg.short_name = model
    return cfg


def list_models() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "models").glob("*.yaml"))


@lru_cache(maxsize=None)
def load_library_config(library: str) -> DictConfig:
    """Library settings from ``configs/libraries/<library>.yaml`` (``name`` added).

    ``library`` is the value of a task's or bundle's ``library`` field.
    """
    path = CONFIG_DIR / "libraries" / f"{library}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no library config {path}")
    cfg = OmegaConf.load(path)
    cfg.name = library
    return cfg


def list_libraries() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "libraries").glob("*.yaml"))


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
