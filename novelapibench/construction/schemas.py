"""Stage-1 records: one entry of an API map, and one candidate of the frozen pool
(``data/pool/candidates.jsonl``). Stages 2-4 use the core ``KnowledgeBundle`` and ``Task``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParameterInfo(BaseModel):
    """One parameter of an introspected signature (``inspect.Parameter``)."""

    name: str
    annotation: str | None = None
    default: str | None = None
    # POSITIONAL_ONLY | POSITIONAL_OR_KEYWORD | VAR_POSITIONAL | KEYWORD_ONLY | VAR_KEYWORD
    kind: str = "POSITIONAL_OR_KEYWORD"


class APIEntry(BaseModel):
    """A public function or class of one library version, as seen by introspection.

    In the frozen pool, ``old_version`` is the release the entry was found new (or modified)
    against, ``None`` when the library had no release before that boundary; ``is_modified``
    marks signature-modified APIs, whose previous signature is kept in ``extra``
    (``old_signature``, ``old_parameters``). ``extra["class_role"]`` records what kind of class
    it is (exception, enum, protocol, typeddict, namedtuple, dataclass, plain_class).
    """

    api_name: str                 # fully qualified name
    kind: str                     # "function" | "class"
    module: str                   # module the entry was found in
    signature: str                # str(inspect.signature(...))
    parameters: list[ParameterInfo] = Field(default_factory=list)
    docstring: str | None = None
    source_file: str | None = None
    library: str
    old_version: str | None = None
    new_version: str
    domain: str = ""
    is_modified: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
