"""Read the repo-root `.env`, with the real environment taking precedence.

Replaces three near-identical copies that had drifted apart on exactly the question of who
wins. They now all use this one, whose rule is the least surprising of the three: **a real
environment variable always beats the file.** (One of the originals ignored the process
environment entirely; another honoured it for three hardcoded keys.) So
`JUDGE_MODEL=... python ...` overrides `.env` everywhere, as you would expect.

`root` is passed in rather than derived from `__file__` — this module lives in an installed
package, so its own location says nothing about where the repo is. Callers already compute
their own repo root; they pass it. Omitting it walks up from the current directory instead.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"\s*([A-Z_][A-Z0-9_]*)=(.*)")
_INTERP = re.compile(r"\$\{(\w+)\}")
# An unquoted value ends where the shell would end it: at whitespace followed by `#`.
_TRAILING_COMMENT = re.compile(r"\s+#.*$")
_LEADING_COMMENT = re.compile(r"\s+#")
# Inside double quotes the shell treats a backslash as an escape only before these characters.
_DQ_ESCAPABLE = '"\\$`'
# Markers of the repo root, in priority order, for the no-argument case.
_ROOT_MARKERS = (".env", "pyproject.toml", ".git")


def find_root(start: Path | None = None) -> Path | None:
    """Nearest ancestor of `start` (default: cwd) that looks like the repo root."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d
    return None


def _double_quoted(raw: str) -> str | None:
    """The value of a `"..."` string as the shell reads it, or None if it never closes."""
    out, i = [], 1
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw) and raw[i + 1] in _DQ_ESCAPABLE:
            out.append(raw[i + 1])
            i += 2
            continue
        if c == '"':
            return "".join(out)
        out.append(c)
        i += 1
    return None


def _value(raw: str) -> str:
    """A `.env` value as `set -a; source .env` would read it, so the two load paths agree.

    `.env.template` ships lines like `MCP_TIMEOUT=120000   # claude-code handshake (ms)`; keeping
    the comment would hand `int()` a string with the comment still attached.
    """
    if _LEADING_COMMENT.match(raw):  # `KEY=   # note` is an empty value
        return ""
    raw = raw.strip()
    # Quoted: the value ends at the closing quote, and a comment may follow. Single quotes are
    # literal; double quotes honour backslash escapes (`"pa\\"ss"` is `pa"ss`).
    if raw[:1] == '"':
        value = _double_quoted(raw)
        if value is not None:
            return value
    elif raw[:1] == "'":
        end = raw.find("'", 1)
        if end != -1:
            return raw[1:end]
    return _TRAILING_COMMENT.sub("", raw)


def load_env(root: Path | None = None) -> dict[str, str]:
    """`.env` values merged under the process environment, with `${VAR}` references expanded.

    Returns the full merged mapping (process environment included), so callers can `.get()` a
    key whether it came from the file or the shell. A missing `.env` is fine — you just get the
    process environment back.
    """
    root = root or find_root()
    parsed: dict[str, str] = {}
    envfile = (root / ".env") if root else None
    if envfile and envfile.is_file():
        for line in envfile.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = _LINE.match(line)
            if m:
                parsed[m.group(1)] = _value(m.group(2))

        # Expand against the file's own values first, then the process environment, so a
        # `.env` line like JUDGE_API_KEY=${OPENAI_API_KEY} resolves either way.
        def expand(v: str) -> str:
            return _INTERP.sub(lambda x: parsed.get(x.group(1), os.environ.get(x.group(1), "")), v)

        parsed = {k: expand(v) for k, v in parsed.items()}
    return {**parsed, **os.environ}
