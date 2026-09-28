"""Knowledge conditions (paper Section 4.2, Appendix D.2) and their rendering.

Twelve conditions supply different subsets of the target API's bundle:

    none, S_name, S, E, M, C, S+M, S+E, S+C, S+M+C, S+E+C, Full

where ``S = S_name + S_param`` and ``Full = S + E + M + C``. Selected components are
rendered in a fixed order (name, parameters, examples, mechanism, implementation) and
joined by blank lines; the prompt places them under a ``# Reference Documentation:`` header.
"""

from __future__ import annotations

from enum import Enum

from novelapibench.schemas import KnowledgeBundle

COMPONENTS = ("S_name", "S_param", "E", "M", "C")


class Condition(str, Enum):
    NONE = "none"
    S_NAME = "S_name"
    S = "S"
    E = "E"
    M = "M"
    C = "C"
    S_M = "S+M"
    S_E = "S+E"
    S_C = "S+C"
    S_M_C = "S+M+C"
    S_E_C = "S+E+C"
    FULL = "Full"

    @property
    def components(self) -> frozenset[str]:
        return _COMPONENTS[self]


_S = {"S_name", "S_param"}
_COMPONENTS: dict[Condition, frozenset[str]] = {
    Condition.NONE: frozenset(),
    Condition.S_NAME: frozenset({"S_name"}),
    Condition.S: frozenset(_S),
    Condition.E: frozenset({"E"}),
    Condition.M: frozenset({"M"}),
    Condition.C: frozenset({"C"}),
    Condition.S_M: frozenset(_S | {"M"}),
    Condition.S_E: frozenset(_S | {"E"}),
    Condition.S_C: frozenset(_S | {"C"}),
    Condition.S_M_C: frozenset(_S | {"M", "C"}),
    Condition.S_E_C: frozenset(_S | {"E", "C"}),
    Condition.FULL: frozenset(COMPONENTS),
}

#: Conditions evaluated on the primary backbone (RQ1) and under retrieval (RQ2).
ALL_CONDITIONS: list[Condition] = list(Condition)
#: The nine conditions evaluated on the other five backbones.
CROSS_BACKBONE_CONDITIONS: list[Condition] = [
    Condition.NONE, Condition.S, Condition.E, Condition.M, Condition.C,
    Condition.S_M, Condition.S_E, Condition.S_C, Condition.FULL,
]


def parse_condition(name: str) -> Condition:
    for c in Condition:
        if name == c.value or name.upper() == c.name:
            return c
    raise ValueError(f"unknown knowledge condition {name!r}; one of {[c.value for c in Condition]}")


def render_knowledge(bundle: KnowledgeBundle, condition: Condition) -> str:
    """The documentation text a condition supplies for ``bundle`` ('' for ``none``)."""
    comps = condition.components
    parts: list[str] = []
    if "S_name" in comps:
        parts.append(f"API: {bundle.s_name}")
    if "S_param" in comps and bundle.s_param:
        parts.append("Parameters:\n" + "\n".join(p.render() for p in bundle.s_param))
    if "E" in comps and bundle.examples:
        parts.append("\n\n".join(f"Example {i + 1}:\n```python\n{ex.code}\n```"
                                 for i, ex in enumerate(bundle.examples)))
    if "M" in comps:
        text = bundle.mechanism.render()
        if text:
            parts.append(f"Conceptual background:\n{text}")
    if "C" in comps and bundle.implementation:
        parts.append(f"Implementation source:\n```python\n{bundle.implementation}\n```")
    return "\n\n".join(parts)
