"""insight_grounding_llm — claim-level precision (P_claim in the paper): the LLM judge.

Tier 2 of insight_grounding (see insight_grounding.py for T1, the deterministic ID floor).
Adapted from DRBench's `factuality` to CliniCARE-Bench: break the report into atomic factual
claims and judge each against the evidence the agent actually retrieved.
DRBench source: github.com/ServiceNow/drbench
(`break_report_to_insights` + `get_factuality_verdict_multi`).

Two adaptations vs DRBench: (1) the *source resolver* — DRBench fetches a cited URL/file;
here the "source" is the MIMIC observation pool from the trace (evidence_pool, shared with
T1), so no live MIMIC and no gold set are needed. (2) the *prompts* below reuse DRBench's
two-step structure but are rewritten for this setting — DRBench's assume web/file citations
and generic report insights; ours target clinical claim types + entity/time attribution
against raw MIMIC rows. Not verbatim DRBench prompt text.

Why an LLM (vs T1's substring check): the discriminative grounding failures are numeric /
clinical claims — a lab value stated for the wrong patient or time, a finding the rows don't
support, a value that's close-but-wrong. A verbatim check can't catch "right number, wrong
entity" (the number substring-matches by chance) and can't read clinical context; the judge
is prompted to verify the value **and** its attribution (patient / time / measurement), not
mere presence.

  grounding = supported / n_claims     (per trial; a claim the retrieved evidence doesn't
                                         support is ungrounded whether it's actively wrong
                                         [contradicted] or simply absent [unverifiable])

Two LLM calls per report: (1) extract atomic claims, (2) judge them against the evidence
(batched; chunked + OR-aggregated when the pool exceeds the model context). Creds come from
the repo-root .env (JUDGE_BASE_URL / JUDGE_API_KEY);
JUDGE_MODEL / --model overrides the default.

WHAT EACH CLAIM IS JUDGED AGAINST — two modes, `--evidence pool|cited`:

  pool  (default; the published numbers): every claim is judged against the WHOLE evidence pool
        the agent retrieved in the run. A claim is supported when the fact appears anywhere in
        what the agent saw, with the right entity/time attribution. This measures retrieved-
        evidence support (faithfulness to the retrieved record): it catches fabrication and
        over-extrapolation, but NOT a correct fact hung on the wrong citation — a claim whose
        citation points at the wrong record still passes if another retrieved row has the fact.
        Reports carry no structured claim→citation markup, which is why this was the metric.
  cited : claim–citation pairs. Extraction also returns, per claim, the citation handles the
        report attaches to it (record / note / order ids, timestamps, tool or table names); the
        resolver maps each handle to the retrieved observations that contain it, and the judge
        sees ONLY those rows. A claim with no handle (beyond the case's own subject_id/hadm_id,
        which every row carries) is `unresolvable` (no citation); one whose handles match no
        retrieved row is `unresolved` (the citation points at nothing the agent retrieved). Both
        count against the strict score. Per claim, `evidence_refs` records exactly which
        observations the judge saw, in both modes.

  grounding (pool)  = supported / n_claims
  grounding (cited) = supported / n_claims          (strict: unresolvable + unresolved count against)
  strict_on_resolvable = supported / n_resolvable    (precision of the citations that exist)
  resolvable_frac   = n_resolvable / n_claims        (how often the report gives a usable citation)

  cd <repo root>
  python benchmark/eval/metrics/grounding/insight_grounding_llm.py \
      --traces benchmark/eval/traces --limit 3 --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clinicare_core import load_env
from clinicare_core.paths import bootstrap

HERE = Path(__file__).resolve().parent

bootstrap(__file__, "eval", "grounding")
from core.trajectory import EXEC_OUTPUT_TOOL, MIMIC_TOOLS, case_scope, load_trajectory  # noqa: E402
from evidence_retrieval_recall import _FROM_RX, TOOL_TABLES, canon  # noqa: E402  (table-name citations)
from insight_grounding import evidence_observations, evidence_pool  # noqa: E402  (T1 shares the loader)

ROOT = HERE.parents[3]  # repo root (grounding → metrics → eval → benchmark → root)
# Sized for a ~128k-token window (~500k chars), leaving headroom for the prompt + claims.
# Judges with a larger window are unaffected — a smaller chunk is always safe, it just costs
# an extra call. Don't raise this to match a specific model: chunk size changes which claims
# share context, so it is not result-neutral.
_CHUNK_CHARS = int(os.environ.get("IG_CHUNK_CHARS", "350000"))
_MAX_RETRIES = int(os.environ.get("IG_MAX_RETRIES", "3"))
_RETRY_BASE_SEC = float(os.environ.get("IG_RETRY_BASE_SEC", "2.0"))


def make_client(env: dict):
    from openai import OpenAI

    base, key = env.get("JUDGE_BASE_URL"), env.get("JUDGE_API_KEY")
    if not base or not key:
        sys.exit("JUDGE_BASE_URL / JUDGE_API_KEY not set (see .env.template).")
    return OpenAI(base_url=base, api_key=key)


def _call(client, model: str, prompt: str) -> str:
    last: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return r.choices[0].message.content or ""
        # retry API/parse errors only; KeyboardInterrupt/SystemExit propagate immediately
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt >= _MAX_RETRIES:
                break
            time.sleep(_RETRY_BASE_SEC * (2 ** (attempt - 1)))
    raise last  # type: ignore[misc]


def _json_array(text: str) -> list:
    """Pull the first JSON array out of a model response (tolerates ```json fences / prose)."""
    text = re.sub(r"```(?:json)?|```", "", text)
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON array in response")


# ── (1) claim extraction ───────────────────────────────────────────────────────────────────
_EXTRACT_PROMPT = """You are extracting atomic factual claims from a clinical decision report \
for grounding verification.

Extract every claim that asserts a SPECIFIC, checkable clinical or quantitative fact drawn \
from the patient's record: lab/vital values, medication names & doses, imaging/procedure \
findings, counts, dates of events, and presence/absence of a clinical event or order.

Rules:
- One atomic claim per item — split compound sentences.
- Keep the entity/context that makes it checkable (which patient/stay, what time, what test).
- EXCLUDE: the final verdict, restatements of the task criteria, methodology/reasoning \
sentences, and claims whose only content is a record ID (those are checked separately).
- Do NOT invent claims; only extract what the report states.

Return ONLY a JSON array of strings, e.g. ["Serum creatinine peaked at 2.3 mg/dL on 2180-09-20", ...].

REPORT:
{report}
"""


def extract_claims(client, model: str, report: str) -> list[str]:
    raw = _call(client, model, _EXTRACT_PROMPT.format(report=report))
    claims = [c.strip() for c in _json_array(raw) if isinstance(c, str) and c.strip()]
    return claims


_DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}")
# Every physical table a tool reads, canonicalised the way the gold names them, so a citation
# like `labevents` resolves to get_lab_timeline rows and `radiology` to get_imaging_studies.
_TABLE_NAMES = {t for tables in TOOL_TABLES.values() for t in canon(tables)}


def _has_token(text: str, key: str) -> bool:
    """`key` as a complete identifier in `text`: 12014968-RR-9 must not match 12014968-RR-99,
    nor 10001217 match 10001217-107's prefix. Identifier characters are [A-Za-z0-9_-]."""
    return re.search(r"(?<![A-Za-z0-9_-])" + re.escape(key) + r"(?![A-Za-z0-9_-])", text) is not None


def _has_day(text: str, day: str) -> bool:
    """`day` (YYYY-MM-DD) as a whole date in `text`; a following time of day is fine."""
    return re.search(r"(?<!\d)" + re.escape(day) + r"(?!\d)", text) is not None


def _tool_tables(obs) -> set[str]:
    """Canonical table names an observation read. run_sql: the FROM/JOIN names in its query,
    canonicalised the same way the gold is (so `discharge_notes` in the SQL becomes `discharge`)."""
    if obs.tool == "run_sql":
        q = str((obs.args or {}).get("query") or (obs.args or {}).get("sql") or "")
        return canon({m.group(1).lower() for m in _FROM_RX.finditer(q)})
    return canon(TOOL_TABLES.get(obs.tool, []))


# Source-type words a report uses as a citation ("per the discharge summary", "EMAR", "POE"),
# mapped to the canonical tables those sources live in. Matched as whole words, case-insensitive.
_SOURCE_WORDS = {
    "discharge summary": {"discharge"}, "discharge narrative": {"discharge"}, "discharge note": {"discharge"},
    "discharge notes": {"discharge"}, "radiology report": {"radiology"}, "radiology note": {"radiology"},
    "radiology notes": {"radiology"}, "radiology": {"radiology"}, "emar": {"emar"}, "mar": {"emar"},
    "poe": {"poe"}, "orders": {"poe"}, "prescriptions": {"prescriptions"}, "pharmacy": {"pharmacy"},
    "labevents": {"labevents"}, "lab timeline": {"labevents"}, "labs": {"labevents"},
    "chartevents": {"chartevents"}, "vitals": {"chartevents", "vitalsign"}, "vital signs": {"chartevents", "vitalsign"},
    "inputevents": {"inputevents"}, "outputevents": {"outputevents"}, "icustays": {"icustays"},
    "icu stays": {"icustays"}, "admissions": {"admissions"}, "transfers": {"transfers"},
    "microbiology": {"microbiologyevents"}, "medrecon": {"medrecon"}, "medication reconciliation": {"medrecon"},
    "procedureevents": {"procedureevents"}, "diagnoses_icd": {"diagnoses_icd"}, "triage": {"triage"},
    "edstays": {"edstays"}, "ed visit": {"edstays"},
}
_TOOL_IN_HANDLE = re.compile(r"(get_[a-z_]+|list_[a-z_]+|search_notes|run_sql)")
_LONG_ID = re.compile(r"\d{6,}(?:-[A-Z]{1,4}-\d+|-\d+)?")           # 12014968, 12014968-RR-9, 10001217-107
# RR-9 / DS-21 without the subject prefix. Only the note types MIMIC-IV-Note has (discharge
# summary, radiology report, radiology addendum): a looser [A-Z]{1,4}-\d+ also matches ICD-9,
# CURB-65 and POD-2, which are not citations and would be scored as citations to a note that was
# never retrieved.
_SHORT_NOTE = re.compile(r"\b(DS|RR|AR)-(\d+)\b")
_FULL_TS = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?")          # 2180-01-30 15:27
_MD_TS = re.compile(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[ T]+(\d{2}:\d{2})")   # 11-12 13:35, 5/3 10:55 (year elided)
_TIME_ONLY = re.compile(r"(?<![\d:])(\d{2}:\d{2})(?![\d:])")            # 20:17 (a window the report itself set)


def _handle_probes(hl: str) -> list[tuple[str, str]]:
    """Decompose one citation handle into (kind, probe) pairs. A handle like
    `mimic_get_lab_timeline lab_name="hemoglobin"` yields ('tool', 'get_lab_timeline');
    `HADM 24440629` yields ('id', '24440629'); `11-12 13:35` yields ('md', '-11-12 13:35');
    `discharge narrative` yields ('table', 'discharge'). Nothing recognised → []."""
    probes: list[tuple[str, str]] = []
    low = hl.lower()
    m = _TOOL_IN_HANDLE.search(low)
    if m and m.group(1) in MIMIC_TOOLS:
        probes.append(("tool", m.group(1)))
    for m in _LONG_ID.finditer(hl):
        probes.append(("id", m.group(0)))
    for m in _SHORT_NOTE.finditer(hl):
        probes.append(("note", f"{m.group(1)}-{m.group(2)}"))
    for m in _FULL_TS.finditer(hl):
        probes.append(("ts", m.group(0)))
    if not any(k == "ts" for k, _ in probes):
        for m in _MD_TS.finditer(hl):
            probes.append(("md", f"-{int(m.group(1)):02d}-{int(m.group(2)):02d} {m.group(3)}"))
        if not any(k == "md" for k, _ in probes):
            for m in _TIME_ONLY.finditer(hl):
                probes.append(("time", m.group(1)))
    for w, tables in _SOURCE_WORDS.items():
        if re.search(r"(?<![a-z_])" + re.escape(w) + r"(?![a-z_])", low):
            probes.extend(("table", t) for t in tables)
    if canon({low}) & _TABLE_NAMES or low in _TABLE_NAMES:
        probes.extend(("table", t) for t in (canon({low}) or {low}))
    return probes


def _probe_hits(kind: str, probe: str, observations: list) -> set[int]:
    out: set[int] = set()
    for i, o in enumerate(observations):
        text = o.text or ""
        argtxt = json.dumps(o.args, default=str)
        if kind == "tool":
            hit = o.tool == probe
        elif kind == "table":
            hit = probe in _tool_tables(o)
        elif kind == "id":
            hit = _has_token(text, probe) or _has_token(argtxt, probe)
        elif kind == "note":
            hit = re.search(r"\d{6,}-" + re.escape(probe) + r"(?![A-Za-z0-9_-])", text + " " + argtxt) is not None
        elif kind == "ts":
            day = probe[:10]
            hit = _has_day(text, day) or _has_day(argtxt, day)
        elif kind == "md":
            hit = probe in text or probe in argtxt
        elif kind == "time":
            hit = re.search(r"\d{4}-\d{2}-\d{2}[ T]" + re.escape(probe), text) is not None
        else:
            hit = False
        if hit:
            out.add(i)
    return out


def resolve_citations(
    cited: list[str], observations: list, case: dict[str, set[str]]
) -> tuple[list[int], list[str], list[str]]:
    """(indices of observations the handles resolve to, handles used, handles dropped).

    Each handle is decomposed into probes (see _handle_probes): tool names resolve to that tool's
    observations; table names and source words ("discharge summary", "EMAR", "POE") to the
    observations of tools that read those tables (run_sql by the tables in its query); record ids
    must occur as complete tokens; a short note id (RR-9) matches with any subject prefix;
    timestamps match on their calendar day, or on month-day-time / time-of-day when the report
    elided the year. A handle that is one of the case's own ids (the task's subject_id / hadm_id /
    stay_id) is dropped: every retrieved row carries those, so it points at nothing. A handle
    with no recognisable probe (a value, a phrase) is not a citation and is dropped too.

    Code-mode traces (bash, exec-scaffold) record a tool call with empty text and the rows in a
    separate `_exec_output` observation. When a resolved call has no text, the exec outputs of
    the same step are added, or every exec output when the step cannot be matched (async exec
    cells return their rows in a later `wait` step)."""
    case_ids = set().union(*case.values()) if case else set()
    used, dropped, hits = [], [], set()
    for h in cited:
        hl = h.strip().strip("`*_[]()")
        kv = re.match(r"^\s*[A-Za-z_]+\s*[:=]\s*(.+?)\s*$", hl)
        if kv and not _TOOL_IN_HANDLE.search(hl.lower()):
            hl = kv.group(1).strip("`*_[]()\"'")
        probes = [(k, v) for k, v in _handle_probes(hl) if not (k == "id" and v in case_ids)]
        if not hl or hl in case_ids or not probes:
            dropped.append(h)
            continue
        # A handle resolves through its MOST SPECIFIC probe tier only: a record id if it has one,
        # else a timestamp, else a tool, else a table / source word. `radiology report
        # 12014968-RR-99` therefore stands or falls on RR-99; the words "radiology report" do not
        # widen it to every radiology row, which would credit a citation to a note never retrieved.
        found: set[int] = set()
        for tier in (("id", "note"), ("ts", "md", "time"), ("tool",), ("table",)):
            tier_probes = [(k, v) for k, v in probes if k in tier]
            if tier_probes:
                for k, v in tier_probes:
                    found |= _probe_hits(k, v, observations)
                break
        (used if found else dropped).append(h)
        hits |= found
    # Follow code-mode calls to their rows.
    exec_idx = [i for i, o in enumerate(observations) if o.tool == EXEC_OUTPUT_TOOL]
    for i in sorted(hits):
        o = observations[i]
        if o.tool != EXEC_OUTPUT_TOOL and not (o.text or "").strip():
            hits |= set(_call_outputs(i, observations, exec_idx))
    return sorted(hits), used, dropped


_EXEC_PENDING = re.compile(r"Script running with cell ID")
_EXEC_DONE = re.compile(r"Script completed")


def _call_outputs(i: int, observations: list, exec_idx: list[int]) -> list[int]:
    """Exec outputs that carry the rows of the code-mode call at index `i`.

    Sync script: the exec output of the same step. Async script: the same-step output is only
    the stub "Script running with cell ID N" and the rows arrive in a later `wait` output, so
    the stub is kept (it is what the agent saw) and the outputs that follow are added up to and
    including the first one that reports "Script completed". When the step has no exec output
    at all the call cannot be placed, and every exec output is returned, as before."""
    step = observations[i].step_id
    same = [j for j in exec_idx if observations[j].step_id == step]
    if not same:
        return exec_idx
    if not all(_EXEC_PENDING.search(observations[j].text or "") for j in same):
        return same
    out = list(same)
    for j in exec_idx:
        if j > max(same):
            out.append(j)
            if _EXEC_DONE.search(observations[j].text or ""):
                break
    return out


def citation_kind(cited: list[str], case: dict[str, set[str]]) -> str:
    """'none' when the claim carries no citation handle at all, 'descriptor' when every handle is
    a case id or an unrecognisable phrase (a value, "the chart"), else 'pointer'. Only a pointer
    that matches nothing retrieved is an `unresolved` citation; the other two are `unresolvable`."""
    if not cited:
        return "none"
    case_ids = set().union(*case.values()) if case else set()
    for h in cited:
        hl = h.strip().strip("`*_[]()")
        kv = re.match(r"^\s*[A-Za-z_]+\s*[:=]\s*(.+?)\s*$", hl)
        if kv and not _TOOL_IN_HANDLE.search(hl.lower()):
            hl = kv.group(1).strip("`*_[]()\"'")
        if hl in case_ids:
            continue
        if any(not (k == "id" and v in case_ids) for k, v in _handle_probes(hl)):
            return "pointer"
    return "descriptor"


# ── (1b) cited mode: which citation belongs to a claim ─────────────────────────────────────
# Decided by rule, not by the extraction model. Asked to copy "the citation attached to each
# claim", the model answers differently on a re-run of the same report (citation coverage moved
# 0.13 per report between runs on a 24-report check); asked only for the NUMBER of the sentence a
# claim came from, with the citation read off that sentence by the rule below, it moved 0.04, and
# what is left is the model listing slightly different claims, not attaching them differently.
#
# The rule. The report is cut into units: a sentence of running text, a list item's sentences, or
# one table row. A claim's citation is every citation handle in its own unit; when its unit has
# none, every handle in the rest of the same block (paragraph, list item or table row), because
# reports routinely name the note or tool once at the head of a bullet and then describe it; when
# the block has none either, the handles of the list item it is nested under. The scope that
# supplied the citation is recorded per claim ("sentence" / "block" / "parent" / "none"), so the
# stricter sentence-only reading can be computed from the same output.
#
# A citation handle is (1) a backtick, bracket or parenthetical span that names a tool, a table or
# source type, a record id, or a full date; (2) outside such spans, a tool name, a MIMIC table
# name, a record id, or a full date. Policy citations ([policy: file.md L1-L9]) are not
# patient-evidence citations and are skipped.
UNITS_VERSION = 1    # segmentation into units; part of the claims cache key
ATTACH_VERSION = 2   # which citation a unit gets (2: nested list items inherit the parent item)

_LIST_ITEM = re.compile(r"^\s*([-*+•]|\d+[.)])\s+")
_ABBREV_END = re.compile(r"\b(?:e\.g|i\.e|vs|Dr|No|approx|Fig|cf|etc|al|incl|est|min|max|hr|hrs)\.$", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?;])[\"')\]*`_]*\s+(?=[A-Z0-9\[(*`\"“])")
_CITATION_ONLY = re.compile(r"^\s*(?:(?:\[[^\]]*\]|\([^)]*\)|`[^`]*`)[\s,;.]*)+$")
_MIN_SENTENCE = 25


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _blocks(report: str) -> list[tuple[str, int | None]]:
    """(text, parent) per block. A block is a heading, a table row, a list item (with its
    continuation lines), or a paragraph; blank lines, rules and table separator rows end a block.
    `parent` is the index of the list item a nested list item (or an indented paragraph inside a
    list) sits under, else None. Headings, table rows and unindented paragraphs end the list."""
    out: list[tuple[str, int | None]] = []
    cur: list[str] = []
    cur_parent: list[int | None] = [None]
    stack: list[tuple[int, int]] = []          # (indent, block index) of the open list items

    def flush() -> None:
        if cur:
            out.append((" ".join(s.strip() for s in cur), cur_parent[0]))
            cur.clear()

    def parent_for(indent: int) -> int | None:
        return next((b for ind, b in reversed(stack) if ind < indent), None)

    for line in report.splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"[-=*_]{3,}", s):
            flush()
        elif s.startswith("#"):
            flush()
            stack.clear()
            out.append((s, None))
        elif s.startswith("|"):
            flush()
            stack.clear()
            if not re.fullmatch(r"[|\s:-]+", s):
                out.append((s, None))
        elif _LIST_ITEM.match(line):
            flush()
            ind = _indent(line)
            while stack and stack[-1][0] >= ind:
                stack.pop()
            cur_parent[0] = stack[-1][1] if stack else None
            stack.append((ind, len(out)))       # this item becomes block len(out) at flush
            cur.append(line)
        else:
            if not cur:                         # a new paragraph
                ind = _indent(line)
                if ind == 0:
                    stack.clear()
                cur_parent[0] = parent_for(ind) if ind else None
            cur.append(line)
    flush()
    return out


def report_blocks(report: str) -> list[str]:
    """Block texts of a markdown report (see _blocks)."""
    return [text for text, _ in _blocks(report)]


def report_units(report: str) -> list[dict]:
    """The report as units: [{"block": i, "text": ...}]. Headings and table rows are one unit;
    other blocks are split into sentences. A fragment shorter than _MIN_SENTENCE characters or
    ending in an abbreviation is joined to what follows, and a fragment that is only a citation
    is joined to the sentence before it."""
    units: list[dict] = []
    for b, (text, parent) in enumerate(_blocks(report)):
        if text.startswith("|") or text.startswith("#"):
            units.append({"block": b, "text": text, "parent": parent})
            continue
        parts: list[str] = []
        buf = ""
        for piece in _SENT_SPLIT.split(text):
            buf = f"{buf} {piece}".strip()
            if _ABBREV_END.search(buf) or len(buf) < _MIN_SENTENCE:
                continue
            parts.append(buf)
            buf = ""
        if buf:
            if parts:
                parts[-1] = f"{parts[-1]} {buf}"
            else:
                parts.append(buf)
        merged: list[str] = []
        for part in parts:
            if merged and _CITATION_ONLY.match(part):
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        units.extend({"block": b, "text": part, "parent": parent} for part in merged)
    return units


_CITE_SPAN = re.compile(r"`[^`\n]+`|\[[^\]\n]+\]|\([^()\n]+\)")
_STRONG_PROBES = {"tool", "id", "note", "ts", "table"}
# MIMIC table names that are not ordinary English words: a source even in running text.
_BARE_TABLE = re.compile(
    r"(?<![A-Za-z_])(chartevents|labevents|inputevents|outputevents|microbiologyevents|procedureevents|"
    r"diagnoses_icd|procedures_icd|drgcodes|edstays|icustays|emar|poe|medrecon|pyxis|vitalsign)(?![A-Za-z_])",
    re.I,
)


def unit_handles(text: str) -> list[str]:
    """Citation handles in one unit, by the rule above. Order of first appearance, no repeats."""
    handles: list[str] = []
    for m in _CITE_SPAN.finditer(text):
        span = m.group(0)
        if re.match(r"[\[(`]\s*policy", span, re.I) or re.search(r"\.md\b", span):
            continue
        inner = span[1:-1].strip()
        if {k for k, _ in _handle_probes(inner)} & _STRONG_PROBES:
            handles.append(inner)
    rest = _CITE_SPAN.sub(" ", text)
    handles += [m.group(0) for m in _TOOL_IN_HANDLE.finditer(rest.lower()) if m.group(1) in MIMIC_TOOLS]
    for rx in (_LONG_ID, _SHORT_NOTE, _FULL_TS, _BARE_TABLE):
        handles += [m.group(0) for m in rx.finditer(rest)]
    return list(dict.fromkeys(handles))


def attach_citation(units: list[dict], i: int) -> tuple[list[str], str]:
    """(handles, scope) for a claim taken from unit `i`: the unit's own handles ("sentence"), else
    the handles of its block ("block"), else those of the list item it is nested under
    ("parent"), else ([], "none")."""
    if not 0 <= i < len(units):
        return [], "none"
    own = unit_handles(units[i]["text"])
    if own:
        return own, "sentence"
    def block_handles(b: int) -> list[str]:
        hs: list[str] = []
        for u in units:
            if u["block"] == b:
                hs += unit_handles(u["text"])
        return list(dict.fromkeys(hs))

    block = block_handles(units[i]["block"])
    if block:
        return block, "block"
    # A nested list item under "1. CT chest (note `…-RR-28`):" is about that note: walk up the
    # enclosing list items, nearest first. Siblings and headings are not inherited.
    parent_of = {u["block"]: u.get("parent") for u in units}
    b, seen = parent_of.get(units[i]["block"]), set()
    while b is not None and b not in seen:
        seen.add(b)
        hs = block_handles(b)
        if hs:
            return hs, "parent"
        b = parent_of.get(b)
    return [], "none"


def pointer_kind(cited: list[str], case: dict[str, set[str]]) -> str:
    """What the citation points with, strongest first: a record id ("record"), a tool or table
    ("source"), only a date or time ("date"), or nothing ("none"). The case's own ids do not
    count, as in resolve_citations."""
    case_ids = set().union(*case.values()) if case else set()
    kinds = {k for h in cited for k, v in _handle_probes(h.strip().strip("`*_[]()")) if not (k == "id" and v in case_ids)}
    if kinds & {"id", "note"}:
        return "record"
    if kinds & {"tool", "table"}:
        return "source"
    if kinds & {"ts", "md", "time"}:
        return "date"
    return "none"


_EXTRACT_UNITS_PROMPT = _EXTRACT_PROMPT.split("Return ONLY")[0] + """\
The report is given below as numbered units: one sentence, list item sentence or table row per \
unit. For EACH claim give the number of the unit it was taken from.

Return ONLY a JSON array of objects, e.g.
[{{"claim": "Serum creatinine peaked at 2.3 mg/dL on 2180-09-20", "unit": 14}}, ...]

REPORT (numbered units):
{report}
"""


def extract_claims_units(client, model: str, units: list[dict]) -> list[dict]:
    """[{"claim": ..., "unit": zero-based unit index, or -1 when the model gave no usable number}]."""
    numbered = "\n".join(f"[{i + 1}] {u['text']}" for i, u in enumerate(units))
    out = []
    for o in _json_array(_call(client, model, _EXTRACT_UNITS_PROMPT.format(report=numbered))):
        if not (isinstance(o, dict) and isinstance(o.get("claim"), str) and o["claim"].strip()):
            continue
        try:
            i = int(o.get("unit")) - 1
        except (TypeError, ValueError):
            i = -1
        out.append({"claim": o["claim"].strip(), "unit": i if 0 <= i < len(units) else -1})
    return out


# Claims are extracted once per report and kept next to the verdict cache, so that re-judging
# (a resolver change, a different evidence mode) scores the same claims instead of a fresh draw.
CLAIMS_CACHE_NAME = "insight_claims_units.json"


def claims_key(report: str, judge_model: str, base_url: str = "") -> dict:
    return {"units_version": UNITS_VERSION, "judge_model": judge_model, "judge_base_url": base_url,
            "prompt_sha": _sha(_EXTRACT_UNITS_PROMPT), "report_sha256": _sha(report)}


def load_or_extract_claims(client, model: str, report: str, base_url: str, cache: Path | None) -> list[dict]:
    key = claims_key(report, model, base_url)
    if cache is not None and cache.exists():
        try:
            d = json.loads(cache.read_text())
            if all(d.get(k) == v for k, v in key.items()) and isinstance(d.get("claims"), list):
                return d["claims"]
        except (OSError, ValueError):
            pass
    claims = extract_claims_units(client, model, report_units(report))
    if cache is not None:
        cache.write_text(json.dumps({**key, "claims": claims}, indent=2))
    return claims


# ── (2) judge claims against retrieved evidence ────────────────────────────────────────────
_JUDGE_PROMPT = """You are verifying whether clinical claims are grounded in the evidence a \
research agent actually retrieved from a patient's medical record.

EVIDENCE (raw rows/notes the agent retrieved from the database):
{evidence}

For EACH numbered claim, decide:
- "supported"   : the evidence contains this fact AND attributes it to the correct entity \
(patient/stay), time, and measurement. Be strict about qualifiers: if the claim specifies a \
time/window/status (e.g. "during the ICU stay", "in-window", "administered"), the evidence \
must show the fact under THAT qualifier. A value that appears but belongs to a different \
patient, a different time, an out-of-window event, or an order-vs-performed mismatch is NOT \
supported.
- "contradicted": the evidence directly conflicts with the claim (different value, or states \
the event did not occur / occurred outside the claimed window).
- "unverifiable": the evidence neither supports nor contradicts it (the fact is absent from \
what was retrieved, or the claim is a clinical judgment the rows can't confirm).

Judge ONLY against the evidence above — not your own medical knowledge.

CLAIMS:
{claims}

Return ONLY a JSON array, one object per claim IN ORDER:
[{{"i": 1, "verdict": "supported|contradicted|unverifiable", "reason": "<brief, cite the value/row>"}}, ...]
"""


def _judge_batch(client, model: str, claims: list[str], evidence: str) -> list[dict]:
    """Grade one small batch of claims against one evidence chunk. Retries if the response
    doesn't cover all indices (long arrays occasionally truncate / re-number) so a dropped
    index becomes a re-roll, not a silent 'no grade' that would deflate the score."""
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    prompt = _JUDGE_PROMPT.format(evidence=evidence, claims=numbered)
    best_by_i: dict[int, dict] = {}
    for _ in range(_MAX_RETRIES):
        try:
            out = _json_array(_call(client, model, prompt))
        except Exception:  # noqa: BLE001  (parse failure → re-roll)
            continue
        for o in out:
            if isinstance(o, dict) and "i" in o:
                try:
                    idx = int(o["i"])
                except (TypeError, ValueError):
                    continue
                # Canonicalize verdict case here so both the _RANK aggregation and score()
                # see lowercase — a judge "Supported" would otherwise rank 0 and score 0.
                if isinstance(o.get("verdict"), str):
                    o["verdict"] = o["verdict"].strip().lower()
                best_by_i.setdefault(idx, o)
        if all((i + 1) in best_by_i for i in range(len(claims))):
            break
    # None for indices the judge never returned (distinct from a real "unverifiable" grade),
    # so cross-chunk aggregation prefers any real grade over an omission.
    return [best_by_i.get(i + 1) for i in range(len(claims))]


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    n = (len(text) + size - 1) // size
    step = len(text) // n
    return [text[i * step : (i + 1) * step if i < n - 1 else len(text)] for i in range(n)]


# precedence when OR-aggregating a claim's verdict across evidence chunks: any chunk that
# supports wins; else any that contradicts; else unverifiable.
_RANK = {"supported": 2, "contradicted": 1, "unverifiable": 0}
# Cap claims per judge call — long JSON arrays truncate mid-response, dropping later indices.
_CLAIM_BATCH = int(os.environ.get("IG_CLAIM_BATCH", "20"))


def judge_claims(client, model: str, claims: list[str], evidence: str) -> list[dict]:
    if not claims:
        return []
    best: list[dict | None] = [None] * len(claims)
    # Each evidence chunk × each small claim-batch is one call; OR-aggregate across chunks
    # (a claim supported by any chunk is supported) so chunking never hides evidence.
    for part in _chunks(evidence, _CHUNK_CHARS):
        for b0 in range(0, len(claims), _CLAIM_BATCH):
            graded = _judge_batch(client, model, claims[b0 : b0 + _CLAIM_BATCH], part)
            for j, g in enumerate(graded):
                k = b0 + j
                if g is None:
                    continue
                if best[k] is None or _RANK.get(g.get("verdict", ""), 0) > _RANK.get(
                    best[k].get("verdict", ""), 0
                ):
                    best[k] = g
    return [b or {"verdict": "unverifiable", "reason": "ungraded (judge omitted)"} for b in best]


def score(claims: list[str], verdicts: list[dict], mode: str = "pool") -> dict:
    """grounding = supported / n_claims. Extraction restricts to *checkable record facts*, so
    a claim the retrieved evidence doesn't support is ungrounded regardless of whether it's
    actively wrong (contradicted) or simply absent (unverifiable) — both count against. The
    3-way breakdown is kept so a low score can be read (fabrication vs. under-retrieval)."""
    counts = {"supported": 0, "contradicted": 0, "unverifiable": 0}
    detail = []
    for c, v in zip(claims, verdicts):
        verd = v.get("verdict", "unverifiable").lower()  # canonical case (defensive)
        counts[verd] = counts.get(verd, 0) + 1
        detail.append({"claim": c, **v} if isinstance(c, str) else {**c, **v})
    grounding = counts["supported"] / len(claims) if claims else None
    out = {"grounding": grounding, "n_claims": len(claims), **counts, "detail": detail}
    # cited mode: the two extra verdict kinds and the derived rates, always (a fully cited
    # report has resolvable_frac 1.0, not a missing field).
    if mode == "cited" or "unresolvable" in counts or "unresolved" in counts:
        counts.setdefault("unresolvable", 0); counts.setdefault("unresolved", 0)
        out.update({k: counts[k] for k in ("unresolvable", "unresolved")})
        n_res = len(claims) - counts.get("unresolvable", 0)
        out["resolvable_frac"] = n_res / len(claims) if claims else None
        out["strict_on_resolvable"] = counts["supported"] / n_res if n_res else None
        # The two components reported in the paper. coverage: share of claims that cite a record,
        # source or date (a citation that matches nothing retrieved still counts as a citation).
        # citation_precision: of those, the share the cited evidence supports. The `_sentence`
        # pair restricts "cites" to a citation in the claim's own sentence.
        cited = [d for d in detail if d.get("citation_kind") == "pointer"]
        sent = [d for d in cited if d.get("cited_scope") == "sentence"]
        sup = lambda ds: sum(1 for d in ds if str(d.get("verdict", "")).lower() == "supported")  # noqa: E731
        out["coverage"] = len(cited) / len(claims) if claims else None
        out["citation_precision"] = sup(cited) / len(cited) if cited else None
        out["coverage_sentence"] = len(sent) / len(claims) if claims else None
        out["citation_precision_sentence"] = sup(sent) / len(sent) if sent else None
    return out


_ARGS_MAX = 2000


def _evidence_block(obs) -> str:
    """One cited observation as the judge sees it: a header naming the tool call and its
    arguments, then the returned text. Without the header a result such as `COUNT(*) = 0`, or
    a medication list filtered to five drug names, cannot be read: the claim "procedureevents
    was empty" is supported by the row only together with the query that produced it. Raw
    command output (code mode) has no call of its own and gets a plain header."""
    if obs.tool == EXEC_OUTPUT_TOOL:
        head = "### output"
    else:
        args = json.dumps(obs.args, default=str, sort_keys=True) if obs.args else "{}"
        if len(args) > _ARGS_MAX:
            args = args[:_ARGS_MAX] + " …"
        head = f"### call {obs.tool} {args}"
    body = (obs.text or "").strip()
    return f"{head}\n{body}" if body else f"{head}\n(rows are in the output that follows)"


def judge_cited(client, model: str, claims: list[dict], observations: list, case: dict) -> list[dict]:
    """Cited mode: each claim is judged against ONLY the observations its citation handles
    resolve to. Claims sharing the same resolved set are batched into one judge call. The
    verdict records `evidence_refs` — the (step_id, tool) of every observation the judge saw."""
    groups: dict[tuple[int, ...], list[int]] = {}
    meta: list[dict] = []
    for k, c in enumerate(claims):
        idx, used, dropped = resolve_citations(c.get("cited", []), observations, case)
        meta.append({"cited_used": used, "cited_dropped": dropped,
                     "evidence_refs": [[observations[i].step_id, observations[i].tool] for i in idx]})
        kind = citation_kind(c.get("cited", []), case)
        meta[k]["citation_kind"] = kind
        if kind == "none":
            meta[k]["verdict"] = "unresolvable"; meta[k]["reason"] = "no citation attached to the claim"
        elif kind == "descriptor":
            meta[k]["verdict"] = "unresolvable"; meta[k]["reason"] = "citation names no record, time or source (descriptor only)"
        elif not idx:
            meta[k]["verdict"] = "unresolved"; meta[k]["reason"] = "citation matches no retrieved record"
        else:
            groups.setdefault(tuple(idx), []).append(k)
    for idx, ks in groups.items():
        evidence = "\n\n".join(_evidence_block(observations[i]) for i in idx)
        graded = judge_claims(client, model, [claims[k]["claim"] for k in ks], evidence)
        for k, g in zip(ks, graded):
            meta[k].update({"verdict": g.get("verdict", "unverifiable"), "reason": g.get("reason", "")})
    return meta


# Per-trial cache file, alongside process.json / policy_support.json. Named for this metric so the
# two LLM grounding stages cannot clobber each other.
CACHE_NAME = "insight_grounding_llm.json"
CACHE_NAME_CITED = "insight_grounding_llm_cited.json"


def cache_name(mode: str) -> str:
    return CACHE_NAME_CITED if mode == "cited" else CACHE_NAME
# Bump when the prompts or the scoring change; every cache written by an older version is stale.
SCORER_VERSION = 2
# Cited mode only: bump when resolve_citations() changes. The resolver decides which rows the
# judge sees, so a resolver change alters verdicts with no prompt or input change to invalidate
# the cache; the version is part of the cited cache key.
RESOLVER_VERSION = 8


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def inputs_key(report: str, pool: str, judge_model: str, base_url: str = "", mode: str = "pool",
               structure: str = "") -> dict:
    """The identity of THIS grading: scorer version, judge model, and a hash of the exact inputs.

    Stamped into every payload and re-checked before a cache hit. Without it, `--judge-model` or an
    edited report/trajectory silently reuses the previous result — reporting and re-persisting
    numbers produced from different inputs, with nothing in the output to reveal it. That is the
    worst failure mode an eval cache can have, so the key is required, not advisory.
    """
    return {
        "scorer_version": SCORER_VERSION,
        "judge_model": judge_model,
        # Grading CONFIG, not just inputs. Both alter results, not merely throughput:
        #   _CHUNK_CHARS moves the evidence-chunk boundaries, and a fact split across a boundary
        #     can be unsupported in both halves even though OR-aggregation recovers most cases;
        #   _CLAIM_BATCH changes which claims share a judge prompt, and an LLM's verdict depends
        #     on the batch it is graded in.
        # Kept as explicit fields rather than folded into the hash so a stale cache says WHY.
        "chunk_chars": _CHUNK_CHARS,
        "claim_batch": _CLAIM_BATCH,
        # The endpoint, because the same model NAME can be a different deployment behind a
        # different proxy. Not a secret, so stored plainly.
        #
        # MUST be the value make_client() actually dialed, i.e. load_env()'s merged mapping —
        # NOT os.environ. load_env returns {**dotenv, **os.environ} without exporting, so a
        # JUDGE_BASE_URL set only in .env reaches the client but is ABSENT from os.environ. An
        # os.environ fallback here would record "" forever: a field that looks like coverage and
        # never invalidates. Callers pass it in; the default is only for direct unit calls.
        "judge_base_url": base_url,
        # The prompts themselves, hashed automatically. Previously these were covered only by a
        # MANUAL SCORER_VERSION bump — edit a prompt, forget to bump, and every cache stays valid
        # against changed grading logic. That is a likelier mistake than any of the above.
        "prompts_sha": _sha(_EXTRACT_PROMPT, _JUDGE_PROMPT)
        if mode == "pool"
        else _sha(_EXTRACT_UNITS_PROMPT, _JUDGE_PROMPT),
        "inputs_sha256": _sha(report, pool),
        # Which evidence each claim is judged against (see module doc). Omitted for pool mode so
        # every cache written before the mode existed stays valid — pool mode IS that behaviour.
        # Cited mode resolves citations through each observation's tool, step and arguments and
        # the case ids, none of which the flattened pool text captures; `structure` is their
        # serialisation (see cited_structure), so a corrected trace re-grades.
        **({"evidence_mode": mode, "resolver_version": RESOLVER_VERSION, "units_version": UNITS_VERSION,
            "attach_version": ATTACH_VERSION, "structure_sha": _sha(structure)}
           if mode != "pool" else {}),
    }


# DELIBERATELY NOT in the key, so the next review pass has an answer rather than a finding:
#   * JUDGE_API_KEY — rotating a key must not invalidate a cache, and "same URL, different key,
#     different deployment" is a routing detail judge_base_url already stands in for. Writing
#     secret-derived material into per-trial files on disk is not worth that corner.
#   * the OpenAI SDK / judge model's own version drift — unobservable from here; that is what
#     SCORER_VERSION and --force are for.
#   * temperature — hardcoded to 0 in _call, not configurable.
# The rule: cover everything reachable through a DOCUMENTED interface (CLI flag, .env.template
# var, env knob read by this module). Beyond that, --force.


def cached_ok(path: Path, expect: dict) -> bool:
    """True only if `path` holds a SUCCESSFUL prior grading OF THE SAME INPUTS.

    Two independent rejections:
      * an `error` payload (judge failure, bad model, expired key) — must be retried, or a
        transient failure caches permanently and every later run reports it forever;
      * a key mismatch — different judge model, different scorer version, or changed
        report/trajectory bytes. A payload predating the key carries no `inputs_sha256` and is
        therefore stale by construction, so it is re-graded once rather than trusted.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if payload.get("error"):
        return False
    return all(payload.get(k) == v for k, v in expect.items())


def load_inputs(tj: Path) -> tuple[str, str] | None:
    """(report_text, evidence_pool_text) for one trial, or None when the report is absent. Local
    reads only — no judge calls — which is what makes input-aware validation affordable for cached
    trials too."""
    report = tj.parents[1] / "verifier" / "report.md"
    if not report.exists():
        return None
    pool, _, _ = evidence_pool(tj.parent)
    return report.read_text(), pool


def load_cited_inputs(tj: Path) -> tuple[list, dict]:
    """(observations, case scope) for cited mode — the per-call evidence and the task's own ids."""
    obs, _, _ = evidence_observations(tj.parent)
    return obs, case_scope(load_trajectory(tj).steps)


def cited_structure(observations: list, case: dict) -> str:
    """What the resolver sees besides the row text: per observation (step, tool, args), plus
    the case ids. Part of the cited cache key."""
    return json.dumps(
        [[o.step_id, o.tool, o.args] for o in observations]
        + [{k: sorted(v) for k, v in sorted(case.items())}],
        default=str, sort_keys=True,
    )


def grade_one(report: str, pool: str, env: dict, model: str, expect: dict,
              mode: str = "pool", observations: list | None = None, case: dict | None = None,
              claims_cache: Path | None = None) -> dict:
    """Score ONE trial from already-loaded inputs. Errors come back IN BAND rather than raising:
    one bad trial must not kill the pool, and `cached_ok` uses the `error` key to force a retry.

    A fresh client per trial — the OpenAI SDK object is cheap to build and this keeps the workers
    from sharing mutable state, matching how process.py hands base_url/api_key to each task.
    """
    try:
        client = make_client(env)
        if mode == "cited":
            units = report_units(report)
            claims = [dict(c) for c in load_or_extract_claims(
                client, model, report, env.get("JUDGE_BASE_URL", ""), claims_cache)]
            for c in claims:
                c["cited"], c["cited_scope"] = attach_citation(units, c["unit"])
                c["pointer"] = pointer_kind(c["cited"], case or {})
            verdicts = judge_cited(client, model, claims, observations or [], case or {})
            return {**expect, **score(claims, verdicts, mode)}
        claims = extract_claims(client, model, report)
        verdicts = judge_claims(client, model, claims, pool)
        # Pool mode judges every claim against everything retrieved; say so per claim.
        for v in verdicts:
            v["evidence_refs"] = "all"
        return {**expect, **score(claims, verdicts)}
    except Exception as exc:  # noqa: BLE001 - in-band so the pool survives
        return {**expect, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=HERE.parents[1] / "traces")
    ap.add_argument(
        "--model", default=None, help="judge model (default JUDGE_MODEL or openai/gpt-5.5)"
    )
    ap.add_argument("--limit", type=int, default=0, help="cap number of trials (0 = all)")
    ap.add_argument("--out", type=Path, default=None, help="write per-claim verdicts JSON here")
    ap.add_argument("--verbose", action="store_true", help="print contradicted/unverifiable claims")
    ap.add_argument(
        "--concurrency", type=int, default=12, help="trials graded in parallel (default 12)"
    )
    ap.add_argument(
        "--force", action="store_true", help=f"re-grade trials that already have {CACHE_NAME}"
    )
    ap.add_argument(
        "--evidence", choices=["pool", "cited"], default="pool",
        help="pool: judge each claim against everything retrieved (published); "
             "cited: judge each claim against the records its citation resolves to",
    )
    ap.add_argument("--trials-from", type=Path, default=None,
                    help="only grade trial dirs listed (one path per line) in this file")
    args = ap.parse_args()
    mode = args.evidence
    cname = cache_name(mode)

    env = load_env()
    # Default matches every other judge call site, and is the model the published grounding
    # numbers were produced at — an unset JUDGE_MODEL must reproduce the paper, not diverge.
    model = args.model or env.get("JUDGE_MODEL") or "openai/gpt-5.5"
    # Provider routing prefixes (openai/…) aren't OpenAI-SDK model
    # names; strip to the bare model when calling the proxy through the OpenAI SDK.
    model = model.split("/")[-1]

    trials = sorted(args.traces.glob("*/*/*/trajectory.json"))
    if args.trials_from:
        keep = {Path(l.strip()).resolve() for l in args.trials_from.read_text().splitlines() if l.strip()}
        trials = [t for t in trials if t.parents[1].resolve() in keep]
    if args.limit:
        trials = trials[: args.limit]
    if not trials:
        sys.exit(f"no trajectories under {args.traces}")

    # Partition before spending anything: a re-run must cost only the trials that are missing or
    # previously errored, and must still report the JOB-WIDE mean by folding the cache back in.
    todo, cached, noreport = [], [], []
    for tj in trials:
        (tj.parents[1] / "verifier").mkdir(parents=True, exist_ok=True)
        loaded = load_inputs(tj)
        if loaded is None:
            noreport.append(tj)
            continue
        report, pool = loaded
        obs, case, structure = None, None, ""
        if mode == "cited":
            obs, case = load_cited_inputs(tj)
            structure = cited_structure(obs, case)
        expect = inputs_key(report, pool, model, env.get("JUDGE_BASE_URL", ""), mode, structure)
        if cached_ok(tj.parents[1] / "verifier" / cname, expect) and not args.force:
            cached.append(tj)
        else:
            todo.append((tj, report, pool, expect, obs, case))
    for tj in noreport:
        print(f"  ! {tj.parts[-3]}: no verifier/report.md", file=sys.stderr)

    print(
        f"judge model: {model}   evidence: {mode}   trials: {len(trials)}   to grade: {len(todo)}   cached: {len(cached)}\n"
    )
    extra = "" if mode == "pool" else f" {'nocit':>5s} {'unres':>5s} {'res%':>5s}"
    print(f"{'run':30s} {'scenario':18s} {'grounded':>8s} {'sup':>4s} {'con':>4s} {'unv':>4s}{extra}")
    rows, rates, dump = [], [], []

    def record(tj: Path, s: dict, from_cache: bool) -> None:
        run, slug = tj.parts[-4], tj.parts[-3]
        if s.get("error"):
            print(f"  ! {slug}: {s['error']}", file=sys.stderr)
            return
        g = f"{s['grounding']:.2f}" if s.get("grounding") is not None else "  -"
        tag = " (cached)" if from_cache else ""
        extra = ""
        if mode != "pool":
            rf = s.get("resolvable_frac")
            extra = (f" {s.get('unresolvable', 0):>5d} {s.get('unresolved', 0):>5d} "
                     f"{(f'{rf:.2f}' if rf is not None else '-'):>5s}")
        print(
            f"  {run[:28]:28s} {slug:18s} {g:>8s} {s['supported']:>4d} "
            f"{s['contradicted']:>4d} {s['unverifiable']:>4d}{extra}{tag}"
        )
        if args.verbose:
            for d in s.get("detail", []):
                if d.get("verdict") != "supported":
                    print(f"      [{d.get('verdict')}] {d['claim']}  — {d.get('reason','')[:120]}")
        if s.get("grounding") is not None:
            rates.append(s["grounding"])
        rows.append((run, slug, s))
        dump.append({"run": run, "scenario": slug, **s})

    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(grade_one, report, pool, env, model, expect, mode, obs, case,
                          tj.parents[1] / "verifier" / CLAIMS_CACHE_NAME): tj
                for tj, report, pool, expect, obs, case in todo
            }
            for n, f in enumerate(as_completed(futs), 1):
                tj = futs[f]
                s = f.result()
                # Write the cache even on error: cached_ok() rejects it, so the next run retries,
                # but the payload is on disk for diagnosis instead of only in a lost stdout.
                (tj.parents[1] / "verifier" / cname).write_text(json.dumps(s, indent=2))
                record(tj, s, from_cache=False)
                if n % 25 == 0:
                    print(f"    {n}/{len(todo)} graded", file=sys.stderr)

    # Cached trials count toward the mean, so a re-run reports the same job-wide number rather
    # than only what it happened to grade this time.
    for tj in cached:
        try:
            record(tj, json.loads((tj.parents[1] / "verifier" / cname).read_text()), True)
        except (OSError, ValueError):
            print(f"  ! {tj.parts[-3]}: unreadable cached {cname}", file=sys.stderr)

    if rates:
        print(f"\nmean grounding over {len(rates)} trials: {sum(rates)/len(rates):.3f}")
    if args.out:
        args.out.write_text(json.dumps(dump, indent=2) + "\n")
        print(f"wrote per-claim verdicts → {args.out}")

    # Found inputs but scored none => every one failed to join (missing report, no gold match).
    # Exit non-zero rather than printing an empty table and exiting 0: a silent 0/N reads as
    # "nothing to score" when it means the join is broken.
    if not rates:
        print(
            f"\n❌ scored 0 of {len(trials)} trace(s) — nothing joined",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
