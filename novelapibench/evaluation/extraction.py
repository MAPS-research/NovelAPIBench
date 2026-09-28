"""Code extraction from a model response (Appendix D.1, "Code extraction").

All backbones and conditions use the same procedure: after removing reasoning traces, take the
longest parseable ``python``-tagged code block, else the longest parseable block with any tag;
else a single unterminated block or an entirely unfenced response that parses. If nothing is
found, the procedure is repeated on the text after the last restated completion instruction
(some adapted models restate the prompt before answering). A response with no extractable code
is an empty completion.
"""

from __future__ import annotations

import ast
import re

# DeepSeek-R1 + Qwen3 thinking traces. Both use <think>...</think>.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
# DeepSeek-R1-Distill's chat template injects the opening <think> tag
# into the prompt, so the model's first emitted token is mid-thinking
# and the response only contains the closing </think>. Strip everything
# up to and including the first </think>.
_TRAILING_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response.

    Idempotent. Safe to call on responses from non-thinking models — they
    have no <think> tags so the input is returned unchanged (modulo
    trailing-whitespace strip).
    """
    if "<think>" not in text and "</think>" not in text:
        return text.strip()
    out = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in out:
        # Lone closing tag (R1-Distill case) — strip prefix.
        out = _TRAILING_THINK_CLOSE_RE.sub("", out, count=1)
    return out.strip()


_FENCE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)\s*\n?(.*?)```", re.DOTALL)
_LEADING_FENCE_RE = re.compile(r"^```[A-Za-z0-9_+\-]*\s*\n?", re.DOTALL)
#: A leading fence that carries *no* language tag. DeepSeek-R1-Distill opens its
#: answer region with exactly this, unpaired, which offsets every later fence
#: pair. A tagged leading fence
#: (```python) is a genuine opener and must never be dropped.
_BARE_LEADING_FENCE_RE = re.compile(r"^```[ \t]*\r?\n")
_PY_LANG_TAGS = {"", "python", "py", "python3", "py3"}
#: Language tags that *explicitly* claim Python. Deliberately excludes the empty
#: tag: an untagged block is where thinking models put their reasoning prose.
_EXPLICIT_PY_TAGS = {"python", "py", "python3", "py3"}
#: Language tags seen on fences in practice. Used only to strip a stray tag
#: line when recovering code from an unpaired-fence response.
_KNOWN_FENCE_TAGS = _PY_LANG_TAGS | {
    "bash", "sh", "shell", "console", "text", "txt", "json", "yaml", "yml",
    "html", "xml", "sql", "diff", "pseudo-code", "pseudocode", "output",
}



def extract_code(text: str) -> str:
    """The completion code contained in a model response ('' when there is none)."""
    code = _select_with_prompt_echo_fallback(text)
    # A completion wrapped as ``def solution():`` keeps only the body.
    if code.startswith("def solution():"):
        return "\n".join(code.split("\n")[1:])
    if "def solution():" in code:
        return code[code.index("def solution():") + len("def solution():"):]
    return code


def _strip_tag_line(body: str) -> str:
    """Drop a bare language-tag first line (``python``, ``bash``, ...).

    Needed because such a line is itself valid Python — ``python\nimport os``
    parses fine — so a parseability check alone would happily accept it and the
    sandbox would then raise ``NameError: name 'python' is not defined``, which
    reads as a model failure rather than an extraction artefact.
    """
    body = (body or "").strip()
    head, _, rest = body.partition("\n")
    if head.strip().lower() in _KNOWN_FENCE_TAGS and rest.strip():
        return rest.strip()
    return body


#: A restated completion instruction (the last line of every task prompt).
_PROMPT_INSTRUCTION_RE = re.compile(r"#\s*Complete the following code[^\n]*\n?")


def _select_with_prompt_echo_fallback(text: str) -> str:
    """``_select_block``, plus recovery of an answer written after a restated prompt.

    Some adapted models (small-data LoRA adapters, the AlphaEdit adapter) answer by first
    copying the prompt — context code, the prose task description and the instruction line —
    and then writing the missing lines, all without fences. The core rules find no parseable
    block in such a response and return ``""``. Only in that case, the text after the last
    restated instruction line is extracted by the same rules. The rule can only turn an empty
    extraction into code, never change a non-empty one.
    """
    code = _select_block(text)
    if code:
        return code
    matches = list(_PROMPT_INSTRUCTION_RE.finditer(text or ""))
    if matches:
        tail = (text or "")[matches[-1].end():]
        if tail.strip():
            return _select_block(tail)
    return ""


def _select_block(text: str) -> str:
    """Recover the answer code block from a response.

    1. Remove ``<think>…</think>`` reasoning (or everything up to a lone ``</think>``).
    2. When the fence count is odd and the text opens with an untagged fence, drop that
       fence (DeepSeek-R1-Distill opens its answer this way, which would otherwise shift the
       pairing of every later fence).
    3. Take the longest ``python``-tagged block that parses;
    4. else the longest block of any tag that parses;
    5. else, for a single unterminated opener, the text after it if it parses;
    6. else, for a response without any fence, the whole response if it parses (backbones
       differ in how often they fence their answer; the parse check keeps prose out);
    7. else the longest segment between fences that parses (an answer started in bare code
       followed by a fenced block);
    8. else ``""``: unparseable text is never executed as a completion.
    """
    text = strip_thinking(text).strip()

    if text.count("```") % 2 == 1:
        m = _BARE_LEADING_FENCE_RE.match(text)
        if m:
            text = text[m.end():]

    blocks = _FENCE_BLOCK_RE.findall(text)
    if blocks:
        tagged = [
            body for tag, body in blocks if tag.lower() in _EXPLICIT_PY_TAGS
        ]
        # `_is_parseable("")` is True — the empty module parses — so an empty
        # body would win as a "candidate" and shadow the real answer.
        parseable_tagged = [
            b for b in (_strip_tag_line(x) for x in tagged) if b and _is_parseable(b)
        ]
        if parseable_tagged:
            return max(parseable_tagged, key=len)
        parseable_any = [
            b for b in (_strip_tag_line(body) for _, body in blocks)
            if b and _is_parseable(b)
        ]
        if parseable_any:
            return max(parseable_any, key=len)
        # Deliberately fall through rather than returning "": an odd fence count
        # makes `_FENCE_BLOCK_RE` pair each *closing* fence with the next
        # *opening* one, so a response that began in bare code yields "blocks"
        # that are really prose gaps while the actual answer sits outside them.

    # No usable paired fence. A single unterminated opener is the common
    # truncation shape; keep its tail only when it is real Python.
    m = _LEADING_FENCE_RE.match(text)
    if m:
        tail = text[m.end():].split("```", 1)[0].strip()
        if tail and _is_parseable(tail):
            return tail

    # Bare answer, no fences anywhere: accept it if - and only if - it is
    # actually Python. See step 6 of the docstring.
    if "```" not in text and text and _is_parseable(text):
        return text

    # Interleaved shape: the model starts answering in bare code and only then
    # opens a fence, leaving an odd fence count that no pairing rule recovers
    # (25% of one qwen2.5-coder cell). Split on the marker and keep the longest
    # segment that is real Python. A bare language-tag line is dropped first,
    # because `python\nimport os` happens to parse and would then execute a
    # NameError-raising bare name.
    segments = [
        s for s in (_strip_tag_line(seg) for seg in text.split("```"))
        if s and _is_parseable(s)
    ]
    if segments:
        return max(segments, key=len)
    return ""


def _is_parseable(code: str) -> bool:
    """``True`` when ``code`` compiles as a Python module.

    Catches ``Exception`` **and** the non-``Exception`` errors CPython's parser
    can raise on pathological model output, rather than the narrow
    ``(SyntaxError, ValueError)``. Observed in the wild: a degenerate
    repetition loop from DeepSeek-R1-Distill makes ``ast.parse`` raise
    ``SystemError: Negative size passed to PyUnicode_New``, and deeply nested
    brackets raise ``RecursionError`` / ``MemoryError``. All of these mean the
    same thing for our purposes — the block is not usable Python — and none of
    them should be allowed to abort a 300-task evaluation.
    """
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError, SystemError):
        return False
    except Exception:
        return False
    return True
