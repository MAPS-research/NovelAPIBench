"""Subprocess sandbox: runs generated code in a library's execution environment with a
timeout, memory and CPU limits, and (where available) PID-namespace isolation."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import json
import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def pid_namespace_available() -> bool:
    """Whether ``unshare -Ur --pid --fork --mount-proc`` works for this user.

    Set ``NOVELAPIBENCH_PID_NAMESPACE=0`` to disable the isolation explicitly.
    """
    if os.environ.get("NOVELAPIBENCH_PID_NAMESPACE") == "0":
        return False
    try:
        r = subprocess.run(["unshare", "-Ur", "--pid", "--fork", "--mount-proc", "true"],
                           capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0

# ---------------------------------------------------------------------------
# Mock imports preamble
# ---------------------------------------------------------------------------
# Real-world code often imports project-specific packages that are not installed
# in the sandbox.  We intercept failed imports and replace them with _SafeMock
# objects so that the code can still run without every dependency installed.
# Never mocks underscore-prefixed internal names (e.g. _winapi).
MOCK_IMPORTS_PREAMBLE = """\
import sys
import types
import importlib.machinery
import builtins as _builtins

_orig_import = _builtins.__import__

class _SafeMock:
    \"\"\"Lightweight mock that supports attribute access, call, iter, arithmetic, and subscript.
    Sets __all__ = [] so 'from x import *' never overwrites real names.\"\"\"
    __all__ = []
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _SafeMock()
    def __getattr__(self, k): return _SafeMock()
    def __iter__(self): return iter([_SafeMock(), _SafeMock()])
    def __repr__(self): return "<_SafeMock>"
    def __bool__(self): return True
    def __len__(self): return 2
    def __contains__(self, item): return True
    def __getitem__(self, k): return _SafeMock()
    def __setitem__(self, k, v): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def __add__(self, o): return _SafeMock()
    def __radd__(self, o): return _SafeMock()
    def __sub__(self, o): return _SafeMock()
    def __rsub__(self, o): return _SafeMock()
    def __mul__(self, o): return _SafeMock()
    def __rmul__(self, o): return _SafeMock()
    def __truediv__(self, o): return _SafeMock()
    def __rtruediv__(self, o): return _SafeMock()
    def __eq__(self, o): return True
    def __ne__(self, o): return False
    def __lt__(self, o): return False
    def __le__(self, o): return True
    def __gt__(self, o): return True
    def __ge__(self, o): return True
    def __mro_entries__(self, bases): return ()
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __index__(self): return 0
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        try:
            import torch as _torch
            return _torch.zeros(())
        except Exception:
            return _SafeMock()
    class __class_getitem__(type):
        def __getitem__(cls, item): return cls

def _make_mock_module(name):
    mod = types.ModuleType(name)
    mod.__all__ = []
    mod.__file__ = f"<mock:{name}>"
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__loader__ = None
    mod.__package__ = name.split(".")[0]
    mod.__getattr__ = lambda k: _SafeMock()
    return mod

def _mock_missing(name, glb=None, loc=None, fromlist=(), level=0):
    _builtins.__import__ = _orig_import
    try:
        return _orig_import(name, glb, loc, fromlist, level)
    except (ImportError, ModuleNotFoundError):
        top = name.split(".")[0]
        if top.startswith("_"):
            raise
        if top not in sys.modules:
            sys.modules[top] = _make_mock_module(top)
        mod = sys.modules[top]
        for i, part in enumerate(name.split(".")[1:], 2):
            full = ".".join(name.split(".")[:i])
            if full not in sys.modules:
                sub = _make_mock_module(full)
                setattr(mod, part, sub)
                sys.modules[full] = sub
                mod = sub
            else:
                mod = sys.modules[full]
        if name not in sys.modules:
            sys.modules[name] = mod
        return sys.modules[name]
    finally:
        _builtins.__import__ = _mock_missing

_builtins.__import__ = _mock_missing
"""

# Wrapper script template: sets RLIMIT_AS and RLIMIT_CPU, then exec's the user code.
_SANDBOX_WRAPPER = textwrap.dedent("""\
    import json, resource, sys, platform

    if platform.system() == "Linux":
        memory_bytes = {memory_mb} * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except ValueError:
            pass

    try:
        resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
    except ValueError:
        pass

    sys.argv = [{code_file!r}] + json.loads({argv_json!r})

    exec(compile(open({code_file!r}).read(), {code_file!r}, 'exec'), {{'__name__': '__main__'}})
""")


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    duration_seconds: float

    @property
    def passed(self) -> bool:
        """True if the code exited successfully."""
        return self.returncode == 0 and not self.timed_out

    @property
    def error_msg(self) -> str | None:
        """Last 500 chars of stderr when failed, else None."""
        if self.passed:
            return None
        return self.stderr.strip()[-500:] if self.stderr.strip() else "execution_failed"


def execute_code(
    code: str,
    timeout: int = 30,
    max_memory_mb: int = 16384,
    env_python: str | None = None,
    extra_env: dict[str, str] | None = None,
    working_dir: str | Path | None = None,
    use_mock_imports: bool = False,
    argv: list[str] | None = None,
    pid_namespace: bool | None = None,
) -> ExecutionResult:
    """Execute Python code in a subprocess with timeout and memory limits.

    Args:
        code: Python source code to execute.
        timeout: Max execution time in seconds.
        max_memory_mb: Virtual memory limit in megabytes.
        env_python: Path to Python interpreter (defaults to sys.executable).
        extra_env: Extra environment variables to set.
        working_dir: Working directory for the subprocess.
        use_mock_imports: If True, prepend MOCK_IMPORTS_PREAMBLE to handle
            missing third-party imports gracefully.
        argv: Optional CLI args to expose to the executed code as ``sys.argv[1:]``.
        pid_namespace: Isolate the child in a user + PID namespace (``unshare``). ``None``
            uses it whenever the host supports unprivileged user namespaces.

    Returns:
        ExecutionResult with stdout, stderr, returncode, timed_out, duration.
    """
    python = env_python or sys.executable
    if pid_namespace is None:
        pid_namespace = pid_namespace_available()

    if use_mock_imports:
        code = MOCK_IMPORTS_PREAMBLE + "\n\n" + code

    # Some evaluated code can leave behind transient files long enough for
    # cleanup to race with subprocess teardown on shared filesystems. The
    # execution result has already been produced at that point, so do not fail
    # the whole evaluation on best-effort tempdir cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        code_file = os.path.join(tmpdir, "_solution.py")
        runner_file = os.path.join(tmpdir, "_runner.py")

        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        runner = _SANDBOX_WRAPPER.format(
            memory_mb=max_memory_mb,
            cpu_limit=timeout * 2,
            code_file=code_file,
            argv_json=json.dumps(argv or []),
        )
        with open(runner_file, "w", encoding="utf-8") as f:
            f.write(runner)

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(var, None)

        # Default cwd to the temp directory so that any files created by the
        # executed code land there instead of polluting the project root.
        cwd = str(working_dir) if working_dir else tmpdir

        start = time.monotonic()
        # The child gets its own process group, so a timeout can kill everything it spawned
        # (servers, workers). With ``pid_namespace`` it additionally runs as pid 1 of a fresh
        # user + PID namespace, so generated code that walks the process table and sends
        # signals cannot reach the evaluator.
        argv0 = [python, runner_file]
        if pid_namespace:
            argv0 = ["unshare", "-Ur", "--pid", "--fork", "--mount-proc", python, runner_file]
        proc = subprocess.Popen(
            argv0,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout=stdout[:4096],
                stderr=stderr[:4096],
                returncode=proc.returncode,
                timed_out=False,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            # Kill the whole process group, not just the direct child, so that
            # any servers / background threads spawned by the candidate code
            # are torn down before we return.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout="",
                stderr="",
                returncode=-1,
                timed_out=True,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                timed_out=False,
                duration_seconds=duration,
            )


def _ensure_test_called(test_code: str) -> str:
    """Normalize LLM-generated test code so that it actually fails on bad candidates.

    Two common failure modes are fixed:

    1. ``def test_execution():`` is defined but never called at the top level.
       The sandbox would exit 0 regardless of candidate quality.
       Fix: append ``test_execution()`` after the definition.

    2. ``try: solution() except SomeError as e: assert ...`` with no ``else``
       clause.  If the solution does *not* raise the expected exception the
       except block is skipped, no assertion runs, and the sandbox exits 0.
       Fix: append ``else: raise AssertionError(...)`` to the try/except.
    """
    if not test_code:
        return test_code

    # --- Fix 1: call test_execution() if defined but not invoked ---
    if "def test_execution(" in test_code:
        top_level_calls = [
            line for line in test_code.splitlines()
            if not line.startswith((" ", "\t"))
            and (line.strip() == "test_execution()" or line.strip().startswith("test_execution()"))
        ]
        if not top_level_calls:
            test_code = test_code.rstrip() + "\n\ntest_execution()\n"

    # --- Fix 2: bare try/except where assertions only live inside except ---
    # Pattern (no indentation prefix on try:):
    #   try:\n      solution()\n  except SomeError...:  \n      assert ...\n  <nothing>
    # We append an else clause that fails if the exception was NOT raised.
    import re
    def _add_else_to_bare_try(code: str) -> str:
        lines = code.splitlines(keepends=True)
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Only match top-level `try:` (no leading whitespace)
            if re.match(r'^try\s*:\s*$', line.rstrip()):
                # Collect the whole try/except block
                block_start = i
                block = [line]
                i += 1
                while i < len(lines):
                    l = lines[i]
                    # except/else/finally at col 0 are continuations of the try block
                    is_continuation = bool(re.match(r'^(except|else|finally)\b', l))
                    # A non-indented, non-empty, non-comment, non-continuation line ends block
                    if l.strip() and not l[0].isspace() and not l.strip().startswith('#') and not is_continuation:
                        break
                    block.append(l)
                    i += 1
                block_str = "".join(block)
                # Only modify if there's an except clause with assert and no else:
                has_except = bool(re.search(r'^except\s', block_str, re.MULTILINE))
                has_assert_in_block = 'assert ' in block_str
                has_else = bool(re.search(r'^else\s*:', block_str, re.MULTILINE))
                if has_except and has_assert_in_block and not has_else:
                    # Find the except clause and extract the exception type for the message
                    m = re.search(r'^except\s+([\w.]+)', block_str, re.MULTILINE)
                    exc_name = m.group(1) if m else "the expected exception"
                    # Inject else clause at the same indentation as except (0 = top level)
                    block_str = block_str.rstrip("\n") + (
                        f"\nelse:\n    raise AssertionError("
                        f"'Expected {exc_name} to be raised but it was not')\n"
                    )
                result.append(block_str)
                continue
            result.append(line)
            i += 1
        return "".join(result)

    test_code = _add_else_to_bare_try(test_code)
    return test_code


def _literal_string_argv(expr: ast.AST) -> list[str] | None:
    """Return a literal argv list when ``expr`` is a list/tuple of strings."""
    if not isinstance(expr, (ast.List, ast.Tuple)):
        return None
    out: list[str] = []
    for elt in expr.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            out.append(elt.value)
        else:
            return None
    return out


def infer_cli_argv(test_code: str) -> list[str] | None:
    """Infer CLI argv from parser-style test code.

    Looks for calls like:
      ``parser.parse_args([...])``
      ``parser.parse_args(args=[...])``
      ``parser.parse_args_into_dataclasses([...])``
      ``parser.parse_args_into_dataclasses(args=[...])``
    and returns the first literal string list found.
    """
    if not test_code.strip():
        return None
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        leaf = None
        if isinstance(func, ast.Attribute):
            leaf = func.attr
        elif isinstance(func, ast.Name):
            leaf = func.id
        if leaf not in {"parse_args", "parse_args_into_dataclasses"}:
            continue

        if node.args:
            argv = _literal_string_argv(node.args[0])
            if argv is not None:
                return argv

        for kw in node.keywords:
            if kw.arg == "args":
                argv = _literal_string_argv(kw.value)
                if argv is not None:
                    return argv
    return None


def execute_test_harness(
    setup_code: str,
    solution_code: str,
    test_code: str,
    timeout: int = 30,
    max_memory_mb: int = 16384,
    env_python: str | None = None,
    use_mock_imports: bool = False,
    argv: list[str] | None = None,
    pid_namespace: bool | None = None,
) -> ExecutionResult:
    """Run setup + solution + test code as a combined execution.

    If ``test_code`` defines ``test_execution()`` without calling it (a common
    LLM harness pattern), the call is appended automatically so the test
    actually runs.

    Args:
        setup_code: Imports and fixture setup.
        solution_code: The candidate solution to test.
        test_code: The test assertions.
        timeout: Execution timeout.
        max_memory_mb: Virtual memory limit in megabytes.
        env_python: Python interpreter path.
        use_mock_imports: If True, prepend MOCK_IMPORTS_PREAMBLE.
        argv: Optional CLI args exposed to candidate code as ``sys.argv[1:]``.

    Returns:
        ExecutionResult.
    """
    test_code = _ensure_test_called(test_code)
    argv = argv if argv is not None else infer_cli_argv(test_code)
    combined = "\n\n".join(
        filter(None, [setup_code.strip(), solution_code.strip(), test_code.strip()])
    )
    return execute_code(
        combined,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
        env_python=env_python,
        use_mock_imports=use_mock_imports,
        argv=argv,
        pid_namespace=pid_namespace,
    )
