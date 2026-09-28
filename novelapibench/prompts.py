"""Evaluation prompts (Appendix D.2, Listing "Evaluation prompt").

The prompt is sent as a single user message through each backbone's own chat template.
"""

from __future__ import annotations

from novelapibench.schemas import Task

#: Last line of every task prompt. Also used by code extraction to recover an answer written
#: after a model restates the prompt.
COMPLETION_INSTRUCTION = (
    "# Complete the following code (output only the missing lines, no explanation):"
)

_INSTRUCTIONS = f"""\
# Output a self-contained completion for the missing region. If `context_code`
# does not already provide a required import, define a needed name, or set up
# required state, include the missing lines in your completion.
#
{COMPLETION_INSTRUCTION}
"""

_NO_KNOWLEDGE = """\
{context_code}

# Task:
{description}

""" + _INSTRUCTIONS

_WITH_KNOWLEDGE = """\
# Reference Documentation:
{knowledge_text}

---

{context_code}

# Task:
{description}

""" + _INSTRUCTIONS


def build_prompt(task: Task, knowledge_text: str | None = None) -> str:
    """The task prompt, with ``knowledge_text`` as reference documentation when given.

    With retrieval, ``knowledge_text`` is the retrieved chunks joined by ``\\n\\n---\\n\\n``.
    """
    if knowledge_text:
        return _WITH_KNOWLEDGE.format(knowledge_text=knowledge_text,
                                      context_code=task.context_code or "",
                                      description=task.description)
    return _NO_KNOWLEDGE.format(context_code=task.context_code or "", description=task.description)
