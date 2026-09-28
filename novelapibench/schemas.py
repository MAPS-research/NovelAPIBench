"""Benchmark records: knowledge bundles and tasks (paper Section 3, Appendix B.1).

A knowledge bundle holds the four components of the paper for one target API:

    S_name  ``s_name``          fully qualified API name
    S_param ``s_param``         parameters (name, type, default, description)
    E       ``examples``        usage examples, each with its sandbox execution status
    M       ``mechanism``       natural-language description of behaviour / design
    C       ``implementation``  implementation source with docstrings removed

``S = S_name + S_param`` and ``Full = S + E + M + C``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Knowledge bundle
# ---------------------------------------------------------------------------


class Parameter(BaseModel):
    """One entry of S_param."""

    name: str
    type: str | None = None
    default: str | None = None
    description: str = ""
    constraints: str | None = None

    @field_validator("default", mode="before")
    @classmethod
    def _stringify_default(cls, v):
        # The extraction model sometimes emits 250 / 0.5 / True instead of strings.
        return v if v is None or isinstance(v, str) else str(v)

    def render(self) -> str:
        line = f"  - {self.name}"
        if self.type:
            line += f" ({self.type})"
        if self.default:
            line += f", default={self.default}"
        if self.description:
            line += f": {self.description}"
        return line


class Example(BaseModel):
    """One entry of E. ``status`` is ``executed`` (ran cleanly in the sandbox), ``static``
    (well-formed but blocked by the environment, e.g. needs weights or network) or ``failed``."""

    code: str
    status: Literal["executed", "static", "failed"] = "executed"
    reason: str = ""


class Mechanism(BaseModel):
    """M: a description of the API's behaviour or design, grounded in an associated paper,
    the implementation source, or the docstring."""

    text: str = ""
    grounding: Literal["paper", "source", "docstring"] = "docstring"
    references: list[str] = Field(default_factory=list)
    source_summary: str | None = None
    changelog_notes: str | None = None

    def render(self) -> str:
        parts = []
        if self.text:
            parts.append(self.text)
        if self.references:
            parts.append("Related work: " + "; ".join(self.references))
        if self.changelog_notes:
            parts.append(f"Changelog notes: {self.changelog_notes}")
        return "\n\n".join(parts)


class KnowledgeBundle(BaseModel):
    api_name: str
    library: str
    domain: str = ""
    s_name: str
    s_param: list[Parameter] = Field(default_factory=list)
    return_type: str | None = None
    examples: list[Example] = Field(default_factory=list)
    mechanism: Mechanism = Field(default_factory=Mechanism)
    implementation: str | None = None
    # Novelty type: signature-modified (True) or newly introduced (False), relative to the
    # release before the boundary; the old signature is kept for modified APIs.
    is_modified: bool = False
    old_signature: str | None = None
    old_parameters: list[dict] | None = None
    requires_file_io: bool = False

    @property
    def signature(self) -> str:
        parts = []
        for p in self.s_param:
            s = p.name
            if p.type:
                s += f": {p.type}"
            if p.default:
                s += f" = {p.default}"
            parts.append(s)
        return "(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TestHarness(BaseModel):
    """Executable checks for a task (Appendix B.3).

    ``setup_code`` installs the target-call monitor; ``execution_test`` asserts that the target
    was called and then runs reference-derived scenario assertions. Evaluation adds the
    call-record check (``novelapibench.evaluation.call_check``). ``shape_type_test`` and
    ``mock_test`` are further scenario layers kept from construction; evaluation does not run them.
    """

    __test__ = False  # not a pytest class

    setup_code: str = ""
    execution_test: str = ""
    shape_type_test: str = ""
    mock_test: str = ""
    target_api: str = ""
    generation_method: str = "execute_then_assert"
    scenarios_succeeded: int = 0
    scenarios_total: int = 0


class Task(BaseModel):
    """A masked-region completion task centred on one target API."""

    task_id: str
    api_name: str
    library: str
    domain: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    description: str
    context_code: str = ""
    masked_region: str = ""
    reference_solution: str
    test_harness: TestHarness = Field(default_factory=TestHarness)
