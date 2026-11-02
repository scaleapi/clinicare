"""Harbor-trace loading + MIMIC observation extraction (grounding Stage 0).

Normalizes the agent harnesses that appear in trial dumps so downstream
metrics don't care which produced a trace:

  - claude-code (opus/sonnet): MIMIC tools named ``mcp__mimic__<tool>``; each
    observation-result ``content`` is a JSON string of
    ``{"type": "text", "text": "<rows-json>"}``.
  - gemini-cli:                MIMIC tools named ``mcp_mimic_<tool>`` (single
    underscores; see MCP_PREFIXES), same content wrapper as claude-code.
  - opencode:                  MIMIC tools named ``mimic_<tool>`` (bare ``mimic_``
    prefix, no ``mcp`` segment); ``content`` is the rows JSON string directly (no
    ``{"type":"text",...}`` wrapper). The normalized ``trajectory.json`` is lossy for
    this harness — some steps carry the ``mimic_`` tool_calls but an empty
    ``observation`` (the row outputs are dropped), so we treat the sibling raw log
    ``agent/opencode.txt`` as authoritative when present (``parse_opencode_log``):
    a line-delimited JSON where each ``type=="tool_use"`` record carries
    ``part.tool`` (``mimic_<tool>``), ``part.state.input`` (args), and
    ``part.state.output`` (the returned rows). Falls back to the ``trajectory.json``
    path when the raw log wasn't fetched.
  - codex (gpt):               bare tool names; ``content`` is
    ``"Wall time: N seconds\\nOutput:\\n[{\"type\":\"text\",\"text\":\"<rows-json>\"}]"``.
  - bash code-mode (``mini-swe-agent``, or any harness whose only tool is ``bash``): NO typed
    MIMIC tool calls either — the agent drives MIMIC from a Python/curl script inside a ``bash``
    command (``mcp_server._run_tool("get_note", {...})``, ``curl $MIMIC_TOOLS_URL/call-tool``,
    ``tools.get_lab_timeline(...)``). ``parse_bash_script`` recovers the calls and the command's
    stdout is pooled under ``_exec_output`` (see the bash block). What the model saw of a long
    output is ``output_head`` + ``output_tail`` (mini-swe-agent elides the middle), and that is
    exactly what enters the pool: the pool is the evidence the agent saw, not the rows the
    sidecar returned.
  - codex exec-scaffold (``codex_exec``): NO typed MIMIC tool calls — MIMIC is
    invoked from a JS script inside an ``exec`` tool call
    (``tools.mcp__mimic__<tool>({..})``), with rows returned inline or via a
    later ``wait`` step. ``parse_exec_script`` recovers the calls and the rows
    are pooled under the ``_exec_output`` sentinel. (See the exec-scaffold block.)

``iter_text_payloads`` unwraps either wrapper down to the tool's row text (JSON
for the first two, a Python ``repr`` for exec-scaffold — hence the literal_eval
fallback). Observations are attributed to their tool call via ``source_call_id``
→ ``tool_call_id`` so a claim can be checked against the *specific* record it
cites, not a global blob (typed harnesses only; the exec scaffold's async cells
decouple call from output, so its rows are pooled, not per-call). Sub-agent
(``Agent``) steps carry only a second-hand summary here (``tool == "_subagent"``);
raw sub-agent tool I/O lives in ``agent/sessions/projects/`` and is not parsed yet.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


# MIMIC MCP tool prefixes across harnesses: claude-code uses `mcp__mimic__<tool>`, gemini-cli
# uses `mcp_mimic_<tool>`, opencode uses `mimic_<tool>` (single underscore, no `mcp` segment);
# codex uses bare names. Longest first so a more specific form is tried before a shorter one
# (`mcp_mimic_get_x` matches `mcp_mimic_`, not `mimic_`, since it starts with `mcp`). (See
# normalize_tool.)
# Tool-name prefixes are OWNED by _template/tests/evidence.py — the hardlink constraint runs one
# way (it may not import eval/, but eval/ may import it). Keep ONE copy: a second list that
# drifts makes systems report zero data-access calls with no error. The literal below is only a
# fallback for a checkout where the import fails, and test_evidence.py asserts the two agree.
def _mcp_prefixes() -> tuple[str, ...]:
    try:
        from clinicare_core.paths import bootstrap

        bootstrap(__file__, "container_tests")
        from evidence import _MCP_PREFIXES as _P2  # type: ignore

        return tuple(_P2)
    except Exception:  # noqa: BLE001
        # Deliberately broad and deliberately silent: this runs at import time, and a literal
        # fallback that is verified equal by test_evidence.py is better than failing to load.
        return ("mcp__mimic__", "mcp_mimic_", "mimic_")


MCP_PREFIXES = _mcp_prefixes()


# The MIMIC *data* tool surface, used to tell patient-data calls apart from harness tools
# (Bash/Write/Agent/WaitForMcpServers/apply_patch/exec_command). Policy tools are deliberately NOT
# here: they read the guideline corpus, not the record, so their output does not belong in the
# patient-evidence pool.
#
# DERIVED FROM THE INSTALLED REGISTRY, not hand-listed, because drift is silent and one-directional:
# a tool this set does not recognize has its rows dropped from `evidence_text`, so a record ID the
# agent genuinely retrieved reads as fabricated by the grounding metrics. The literal below is the
# offline fallback only (same pattern as _template/tests/evidence.py, which must be stdlib-only);
# keep it in step, but the registry is the source of truth whenever mimic_tools is importable.
_FALLBACK_MIMIC_TOOLS = frozenset(
    {
        "get_patient_demographics",
        "list_admissions",
        "get_admission_details",
        "list_ed_visits",
        "get_ed_visit_details",
        "list_icu_stays",
        "get_icu_stay_details",
        "get_lab_timeline",
        "get_vitals_timeline",
        "get_medication_administered",
        "get_medication_prescribed",
        "get_medication_reconciliation",
        "get_diagnosis_history",
        "get_procedures",
        "get_imaging_studies",
        "get_microbiology_results",
        "get_fluid_balance",
        "list_notes",
        "get_note",
        "search_notes",
        "run_sql",
        "list_orders",
        "get_chart_timeline",
        "get_data_availability",
        "get_icu_administrations",
    }
)


_FALLBACK_POLICY_TOOLS = frozenset({"list_policy_docs", "search_policy", "read_policy"})


def _registry_tool_sets() -> tuple[frozenset[str], frozenset[str], bool]:
    """(patient-data tools, policy tools, registry_loaded). Falls back to the literals above."""
    try:
        from mimic_tools.tool_registry import POLICY_TOOLS, TOOLS

        return frozenset(set(TOOLS) - set(POLICY_TOOLS)), frozenset(POLICY_TOOLS), True
    except Exception:  # noqa: BLE001 — offline use; recognition falls back, never empties
        return _FALLBACK_MIMIC_TOOLS, _FALLBACK_POLICY_TOOLS, False


MIMIC_TOOLS, POLICY_TOOLS, REGISTRY_LOADED = _registry_tool_sets()
# The whole tool surface. Consumers that count "data-access work" without separating patient
# data from corpus reads (score_clinicare's `data` column) want this, not MIMIC_TOOLS alone.
ALL_TOOLS = MIMIC_TOOLS | POLICY_TOOLS


def normalize_tool(name: str | None) -> str:
    """Strip the MCP prefix (claude-code ``mcp__mimic__`` / gemini-cli ``mcp_mimic_``); codex
    names are already bare."""
    name = name or ""
    for p in MCP_PREFIXES:
        if name.startswith(p):
            return name[len(p) :]
    return name


def is_mimic_tool(name: str | None) -> bool:
    return normalize_tool(name) in MIMIC_TOOLS


# ── codex exec-scaffold support ───────────────────────────────────────────────────────────
# The `codex_exec` originator does NOT emit typed MIMIC tool calls. Instead it runs a JS
# script inside an `exec` tool call, invoking MIMIC tools as
# ``tools.mcp__mimic__<tool>({..args..})`` (often batched in a ``Promise.all([...])``); the
# returned rows arrive either inline in the same step's output or — for async cells — in a
# later ``wait`` step's output. We recover MIMIC tool calls by parsing the script text and
# pool the returned rows (any exec/wait output text) via a sentinel-tool observation so the
# grounding/retrieval metrics work on these traces too. See EXEC_OUTPUT_TOOL below.
EXEC_OUTPUT_TOOL = "_exec_output"  # sentinel: pooled exec/wait row text, not a MIMIC call
_EXEC_MIMIC_REF = re.compile(r"mcp__mimic__(\w+)")
# JS string literals in an exec script — backtick template, double-, or single-quoted (each
# body-class excludes its own quote and honors backslash escapes, incl. escaped backticks). Used
# to recover run_sql queries however they're passed (``query:"..."``, ``const q=`...` ``, or a
# ``qs=[`...`,`...`]`` array .map'd into run_sql). Since evidence_retrieval_recall unions tables
# across the trace, we attribute each to run_sql without binding it to a specific call. run_sql
# is the only MIMIC tool taking a query, so SQL literals belong to it; the text feeds the
# metric's FROM/JOIN table parse.
_EXEC_STR_LITERAL = re.compile(
    r"`((?:[^`\\]|\\.)*)`" r'|"((?:[^"\\]|\\.)*)"' r"|'((?:[^'\\]|\\.)*)'",
    re.S,
)
# A literal counts as SQL only if it has BOTH a SELECT and a FROM<word> — so prose like
# "Calculated from baseline values" or "Missing FROM clause" can't masquerade as a query and
# spawn a spurious run_sql observation. (Every run_sql query in the runs is a SELECT/CTE.)
_SQL_FROM = re.compile(r"\bfrom\s+[a-z_]", re.I)
_SQL_SELECT = re.compile(r"\bselect\b", re.I)


def _sql_literals(script: str) -> list[str]:
    """Every string/template literal in the script that looks like a SQL query (SELECT + FROM)."""
    out: list[str] = []
    for m in _EXEC_STR_LITERAL.finditer(script or ""):
        s = m.group(1) or m.group(2) or m.group(3) or ""
        if _SQL_SELECT.search(s) and _SQL_FROM.search(s):
            out.append(s)
    return out


def parse_exec_script(script: str, step_id: int | None) -> list[Observation]:
    """Recover MIMIC tool calls from one exec-scaffold JS ``input`` script.

    Returns tool Observations with empty ``text`` (the returned rows are pooled separately
    from the exec/wait *output*, which the script text doesn't contain). run_sql carries its
    query in ``args`` so table attribution still works; other tools carry ``{}`` (their tables
    come from the static tool→table map). A tool the script references but that errored at
    run time is still counted as an attempt — a minor over-count vs the typed path, which
    only sees calls that returned."""
    out: list[Observation] = []
    tools = set(_EXEC_MIMIC_REF.findall(script or ""))
    for name in sorted(tools):
        if name == "run_sql" or name not in MIMIC_TOOLS:
            continue  # run_sql handled below (needs its query); non-data tools skipped
        out.append(Observation(step_id=step_id, tool=name, args={}, text=""))
    if "run_sql" in tools:
        queries = _sql_literals(script)
        if queries:
            for q in queries:
                out.append(Observation(step_id=step_id, tool="run_sql", args={"query": q}, text=""))
        else:
            # run_sql called with a query built dynamically (no recoverable literal) — record
            # the call so it is not silently dropped, though its tables can't be parsed.
            out.append(Observation(step_id=step_id, tool="run_sql", args={}, text=""))
    return out


# ── bash code-mode support (mini-swe-agent) ─────────────────────────────────────────────────
# mini-swe-agent exposes ONE tool, `bash`, and no MCP client. Its runs reach MIMIC by scripting
# the in-container `mimic_tools` package or the sidecar's HTTP API from inside a bash command.
# Measured on the harness-ablation dump (148 trials): 147 trials returned patient rows this way;
# the trajectory-typed path sees zero MIMIC calls for all of them. A tool counts as *called* in a
# step only when (a) its name appears in call position or as a string literal — `get_note(`,
# `"get_note"`, `'get_note'` — which excludes the agent merely listing tool names in a plan, and
# (b) the step's command exited 0 with non-empty output, so a script that crashed before reaching
# the sidecar credits nothing. (a) can still over-count a tool named in a dict the script never
# sends; that is the same minor over-count the exec-scaffold parser accepts, and it affects
# retrieval coverage (which unions tables) far less than it would a per-call metric.
# Three shapes count: the name as a whole string literal (`_run_tool("get_note", ...)`,
# `{"name": "get_note"}`), the name in call position (`tools.get_note(...)`), or the name as the
# first argument of a helper script the agent wrote (`python3 call_mimic.py list_notes
# subject_id=...`). A name inside a longer string (`print("next: get_lab_timeline")`) or in a
# comment matches none of them.
_BASH_TOOL_REF = re.compile(
    r"""["'](get_\w+|list_\w+|search_notes|run_sql)["']"""
    r"""|\b(get_\w+|list_\w+|search_notes|run_sql)\s*\("""
    r"""|\.py\s+(get_\w+|list_\w+|search_notes|run_sql)\b"""
)
# Scope ids the script passes, however it passes them: `subject_id=1`, `"subject_id": 1`,
# `subject_id = 1`, `WHERE subject_id = 1`. Only ids that are plausibly arguments (an
# 8-digit MIMIC id) are kept.
_SCOPE_ARG = {
    k: re.compile(rf"""\b{k}\b["']?\s*[:=]\s*["']?(\d{{6,}})""")
    for k in ("subject_id", "hadm_id", "stay_id")
}
# `subject_id IN (1, 2)` in SQL.
_SCOPE_IN = {
    k: re.compile(rf"""\b{k}\b\s+in\s*\(([^)]*)\)""", re.I) for k in ("subject_id", "hadm_id", "stay_id")
}


def _bash_output_text(content) -> tuple[int | None, str]:
    """(returncode, text the model saw) from one bash observation. mini-swe-agent serializes the
    result as JSON ``{"returncode", "output"}`` or, when the output was too long,
    ``{"returncode", "output_head", "output_tail", "elided_chars", "warning"}``."""
    d = None
    if isinstance(content, str):
        try:
            d = json.loads(content)
        except Exception:
            return None, content
    elif isinstance(content, dict):
        d = content
    if not isinstance(d, dict):
        return None, ""
    rc = d.get("returncode")
    text = d.get("output")
    if text is None:
        text = "\n".join(str(d.get(k) or "") for k in ("output_head", "output_tail"))
    return (int(rc) if isinstance(rc, int) else None), str(text or "")


def parse_bash_script(command: str, step_id: int | None) -> list[Observation]:
    """Recover MIMIC tool calls referenced in one bash command (see the block comment above).
    Returns tool Observations with empty ``text``; the command's stdout is pooled separately by
    ``load_trajectory`` under EXEC_OUTPUT_TOOL. ``args`` carries every scope id the script names
    (subject_id / hadm_id / stay_id) so the retrieval metric can check patient scope, and run_sql
    carries each recoverable SQL literal as ``query``."""
    out: list[Observation] = []
    names = {m.group(1) or m.group(2) or m.group(3) for m in _BASH_TOOL_REF.finditer(command or "")} & MIMIC_TOOLS
    if not names:
        return out
    args: dict = {}
    for k, rx in _SCOPE_ARG.items():
        ids = sorted({m.group(1) for m in rx.finditer(command)})
        if ids:
            args[k] = ids[0] if len(ids) == 1 else ids
    for name in sorted(names):
        if name == "run_sql":
            queries = _sql_literals(command)
            for q in queries or [None]:
                a = dict(args)
                if q is not None:
                    a["query"] = q
                out.append(Observation(step_id=step_id, tool="run_sql", args=a, text=""))
        else:
            out.append(Observation(step_id=step_id, tool=name, args=dict(args), text=""))
    return out


# ── case scope: which patient / encounter the task is about ─────────────────────────────────
_CASE_RX = {
    k: re.compile(rf"""\b{k}\b\s*[=:]?\s*(\d{{6,}})""") for k in ("subject_id", "hadm_id", "stay_id")
}


def case_scope(steps) -> dict[str, set[str]]:
    """{"subject_id": {...}, "hadm_id": {...}, "stay_id": {...}} named by the task instruction.

    The instruction is the first ``source == "user"`` step; the case sentence precedes the
    ``---`` that opens the environment boilerplate (which names no ids). Every scenario names
    the patient (``subject_id=12014968`` or ``For subject_id 16474066``); most name the index
    admission; ED scenarios name a stay. An empty set means the instruction names none, and a
    scope check must then abstain rather than flag."""
    msg = next((s.get("message") or "" for s in steps if s.get("source") == "user"), "")
    head = msg.split("\n---", 1)[0]
    return {k: set(rx.findall(head)) for k, rx in _CASE_RX.items()}


def call_scope(obs: Observation) -> dict[str, set[str]]:
    """Scope ids one MIMIC call names, from its typed arguments (``{"subject_id": 10001217}``) or,
    for run_sql / code-mode scripts, from ``subject_id = N`` / ``subject_id IN (...)`` in the query
    text. Ids are returned as strings so typed ints and regex captures compare equal."""
    out: dict[str, set[str]] = {k: set() for k in _SCOPE_ARG}
    args = obs.args if isinstance(obs.args, dict) else {}
    for k in out:
        v = args.get(k)
        if isinstance(v, (int, str)) and str(v).strip().isdigit():
            out[k].add(str(v).strip())
        elif isinstance(v, (list, tuple)):
            out[k] |= {str(x).strip() for x in v if str(x).strip().isdigit()}
    q = args.get("query") or args.get("sql") or ""
    if isinstance(q, str) and q:
        for k in out:
            out[k] |= {m.group(1) for m in _SCOPE_ARG[k].finditer(q)}
            for m in _SCOPE_IN[k].finditer(q):
                out[k] |= {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip().strip("'\"").isdigit()}
    return out


# ── opencode raw-log support ────────────────────────────────────────────────────────────────
# The opencode harness's normalized trajectory.json is lossy: some steps keep the `mimic_`
# tool_calls but an empty `observation` (row outputs dropped). The raw `agent/opencode.txt`
# is complete — line-delimited JSON, one event per line; a `type=="tool_use"` event carries
# the call and its output together. We treat it as authoritative for opencode when fetched.
def parse_opencode_log(path) -> list[Observation]:
    """Recover MIMIC observations from an opencode raw log (agent/opencode.txt).

    Each `type=="tool_use"` line's `part` holds `tool` (``mimic_<tool>``), `state.input`
    (args) and `state.output` (the returned rows, usually a JSON string). Non-MIMIC tools
    (bash/write/read/todowrite/...) are skipped, as are calls with ``state.status=="error"``
    (MIMIC MCP timeouts/failures return no rows — they retrieved nothing, so they must not
    count toward retrieval or enter the evidence pool). step_id is a running index (opencode
    has no step numbering); the metrics union / pool observations, so ordering is all that
    matters."""
    out: list[Observation] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line or '"tool_use"' not in line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") != "tool_use":
            continue
        part = rec.get("part") or {}
        name = part.get("tool")
        if not is_mimic_tool(name):
            continue
        state = part.get("state") or {}
        if state.get("status") == "error":
            continue
        inp = state.get("input")
        payload = state.get("output")
        text = payload if isinstance(payload, str) else (json.dumps(payload) if payload else "")
        out.append(
            Observation(
                step_id=i,
                tool=normalize_tool(name),
                args=inp if isinstance(inp, dict) else {},
                text=text,
            )
        )
    return out


def iter_text_payloads(obj):
    """Yield every text string buried in a nested observation ``content``.

    Recurses through dicts/lists and re-parses serialized strings, handling the
    codex ``"Wall time: ..\\nOutput:\\n[...]"`` prefix before the array. Tries
    ``json.loads`` first, then ``ast.literal_eval`` — the codex *exec-scaffold*
    harness serializes its ``Output`` block as a Python ``repr`` of a list of
    ``{'type': 'input_text', 'text': ...}`` dicts (single-quoted, so not JSON);
    literal_eval unwraps it to the same nested tool-result text the JSON path
    yields for the other harnesses.
    """
    if isinstance(obj, str):
        yield obj
        s = obj.strip()
        m = re.search(r"Output:\s*(\[.*|\{.*)", s, re.S)
        if m:
            s = m.group(1)
        if s[:1] in "[{":
            parsed = None
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None
            if parsed is not None:
                yield from iter_text_payloads(parsed)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_text_payloads(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_text_payloads(v)


@dataclass
class Observation:
    """One MIMIC tool result the agent saw, attributed to its call."""

    step_id: int | None
    tool: str  # normalized tool name, or "_subagent" for Agent summaries
    args: dict  # the tool-call arguments
    text: str  # unwrapped payload text (the returned rows)


@dataclass
class Trajectory:
    path: Path
    agent: str  # "claude-code" | "codex" | ...
    model: str
    steps: list
    used_subagent: bool
    observations: list  # list[Observation], MIMIC tools (+ subagent summaries) only

    @property
    def evidence_text(self) -> str:
        """All MIMIC observation text concatenated (the grounding evidence pool)."""
        return "\n".join(o.text for o in self.observations)

    def observations_for(self, tool: str) -> list[Observation]:
        return [o for o in self.observations if o.tool == normalize_tool(tool)]


def subagent_observations(trial_dir) -> list[Observation]:
    """MIMIC tool calls made *inside* sub-agents, recovered from claude-code's raw session
    JSONL (``subagents/*.jsonl`` next to the trace) — which the normalized trajectory.json
    omits (it only records the parent ``Agent`` call + its summary).

    tool_use blocks only (``name`` + ``input``); the tool *result* text is not captured
    here, so ``Observation.text`` is "" — enough for evidence_retrieval_recall (needs the call, not
    the rows). A factuality use would additionally need the tool_result records.
    Returns [] when the trace dump contains no subagent jsonls.
    """
    out: list[Observation] = []
    for jf in sorted(Path(trial_dir).glob("subagents/*.jsonl")):
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            content = (rec.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if (
                    isinstance(blk, dict)
                    and blk.get("type") == "tool_use"
                    and is_mimic_tool(blk.get("name"))
                ):
                    out.append(
                        Observation(
                            step_id=None,
                            tool=normalize_tool(blk["name"]),
                            args=blk.get("input") or {},
                            text="",
                        )
                    )
    return out


def load_trajectory(path) -> Trajectory:
    """Parse a trial ``agent/trajectory.json`` into a normalized Trajectory."""
    path = Path(path)
    t = json.loads(path.read_text())
    agent = t.get("agent") or {}
    steps = t.get("steps", [])

    observations: list[Observation] = []
    used_subagent = False

    for s in steps:
        calls = {tc.get("tool_call_id"): tc for tc in s.get("tool_calls", [])}
        if any(tc.get("function_name") == "Agent" for tc in s.get("tool_calls", [])):
            used_subagent = True

        # exec-scaffold: recover MIMIC calls from each exec script's text (the returned rows
        # are pooled from the exec/wait output below, not present in the script).
        for tc in s.get("tool_calls", []):
            if tc.get("function_name") == "exec":
                script = (tc.get("arguments") or {}).get("input", "")
                observations.extend(parse_exec_script(script, s.get("step_id")))

        # bash code-mode (mini-swe-agent): MIMIC calls live inside the bash command's script and
        # the rows come back as the command's stdout. Credit the referenced tools only when the
        # command succeeded and printed something. EVERY successful command's stdout is pooled as
        # evidence, not only the steps that name a tool: these agents save tool output to files
        # and read it back in later steps (`cat out.json`, `python3 helper.py ...`), and the pool
        # is defined as what the agent saw. Measured on the harness-ablation arm, all 28 record ids
        # flagged as fabricated under tool-step-only pooling were in the stdout of such steps. The
        # exec-scaffold path pools every exec/wait output for the same reason. The cost is that an
        # id the agent itself echoes (`echo "subject_id=..."`) counts as seen; that is a case id
        # from the instruction in practice, and the exec path already accepts the same.
        bash_calls = {
            tc.get("tool_call_id"): tc
            for tc in s.get("tool_calls", [])
            if tc.get("function_name") == "bash"
        }
        if bash_calls:
            for item in (s.get("observation", {}) or {}).get("results", []) or []:
                tc = bash_calls.get(item.get("source_call_id"))
                # mini-swe-agent's results carry no source_call_id; with one bash call per step
                # (its protocol) the pairing is unambiguous.
                if tc is None and item.get("source_call_id") is None and len(bash_calls) == 1:
                    tc = next(iter(bash_calls.values()))
                if tc is None:
                    continue
                rc, text = _bash_output_text(item.get("content"))
                if rc not in (0, None) or not text.strip():
                    continue
                cmd = (tc.get("arguments") or {}).get("command", "") or ""
                observations.extend(parse_bash_script(cmd, s.get("step_id")))
                observations.append(
                    Observation(step_id=s.get("step_id"), tool=EXEC_OUTPUT_TOOL, args={}, text=text)
                )

        for item in (s.get("observation", {}) or {}).get("results", []) or []:
            tc = calls.get(item.get("source_call_id"))
            fn = tc.get("function_name") if tc else None
            text = "\n".join(iter_text_payloads(item.get("content")))
            if is_mimic_tool(fn):
                observations.append(
                    Observation(
                        step_id=s.get("step_id"),
                        tool=normalize_tool(fn),
                        args=tc.get("arguments") or {},
                        text=text,
                    )
                )
            elif fn == "Agent":
                observations.append(
                    Observation(
                        step_id=s.get("step_id"),
                        tool="_subagent",
                        args={},
                        text=text,
                    )
                )
            elif fn in ("exec", "wait") and text.strip():
                # exec-scaffold row output (inline for sync scripts, or the async `wait`
                # poll). Pooled under a sentinel tool so it feeds the grounding evidence
                # pool (evidence_text) without being mistaken for a MIMIC data call by the
                # retrieval metric.
                observations.append(
                    Observation(
                        step_id=s.get("step_id"),
                        tool=EXEC_OUTPUT_TOOL,
                        args={},
                        text=text,
                    )
                )

    # opencode: the raw log is complete + authoritative where trajectory.json is lossy (some
    # steps keep the mimic_ calls but drop their outputs, and errored calls appear as
    # output-less observations that would over-credit retrieval). When the log is present, it
    # fully replaces the trajectory-derived observations — even if it yields none (a trial whose
    # every MIMIC call errored genuinely retrieved nothing).
    if agent.get("name") == "opencode":
        oc_log = path.parent / "opencode.txt"
        if oc_log.exists():
            observations = parse_opencode_log(oc_log)

    return Trajectory(
        path=path,
        agent=agent.get("name", ""),
        model=agent.get("model_name", ""),
        steps=steps,
        used_subagent=used_subagent,
        observations=observations,
    )
