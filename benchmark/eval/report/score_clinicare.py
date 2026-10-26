"""Aggregate CliniCare-Bench job outputs into the NxM (models × metrics) table.

The ONE place runs are scored. Point it at one or more harbor job dirs (globs OK);
it discovers the model per job, aggregates the canonical per-trial artifacts, and
emits a markdown table plus a CSV for downstream plots.

    uv run python benchmark/eval/report/score_clinicare.py jobs/<job>
    uv run python benchmark/eval/report/score_clinicare.py jobs/<job> --csv out.csv

Per trial it reads:
  verifier/reward.json    -> outcome_score, outcome_label_parsed, the deterministic grounding keys
  verifier/process.json   -> process_pct (scored POST-HOC by eval/metrics/process.py: the in-trial
                             verifier is deterministic-only, so reward.json carries no process axis)
  verifier/evidence.jsonl -> the data-call count, correct across every harness
  verifier/rubrics.json   -> ground_truth_label, scenario_id
  agent/trajectory.json   -> final_metrics (total_steps, *_tokens, total_cost_usd) + steps[] (tool-calls)
  result.json             -> cost/tokens + agent-phase wall clock

Design notes:
  - Row identity comes from config.json (agent import-path/name + model), not the job-name string.
  - The AMB-no-rule exclusion set is computed from _scenarios (a scenario with an EMPTY
    indeterminate_medical_ambiguity_criteria that nonetheless has >=1 gold AMB patient); those
    gold-AMB trials are ungradeable by their own spec and dropped from the "clean" accuracy.
  - Turns are NOT comparable across harnesses (a codex step batches many tool calls; gemini ~1/turn).
    Compare turns within a harness; use $ / work-tok across. work-tok = prompt - cached + completion.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys
import tomllib
from collections import defaultdict
from datetime import datetime

from clinicare_core import load_full, verifier_failed
from clinicare_core.paths import bootstrap

bootstrap(__file__, "eval")
from core.trajectory import ALL_TOOLS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.abspath(os.path.join(HERE, "..", ".."))  # eval/report → eval → benchmark
BASE = os.path.abspath(os.path.join(BENCHMARK, ".."))  # repo root; job task paths resolve from it
SCENARIOS_DIR = os.path.join(BENCHMARK, "_scenarios")

AMB = "INDETERMINATE_MEDICAL_AMBIGUITY"
LACK = "INDETERMINATE_LACK_OF_DATA"
CLASSES = ["YES", "NO", LACK, AMB]
SHORT = {"YES": "YES", "NO": "NO", LACK: "lack", AMB: "amb"}
_CONF_RE = re.compile(r"#+\s*Confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.I)

# MIMIC data + policy tools = the actual data-access work. `tools` (total) mixes these with
# harness-specific overhead (planning/file/web/shell), so it's NOT comparable cross-harness;
# `data` is. Names vary by harness (bare `run_sql` / `mcp__mimic__X` / `mcp_mimic_X`), which the
# prefix strip below handles.
#
# Derived from the tool registry (core.trajectory owns that derivation, with its own offline
# fallback), not hand-listed: a tool missing from this set is counted as harness overhead, so the
# `data` column would silently under-report rather than error. Harness-side tools are excluded by
# construction — execute_python is compute, not a data query, and web_search is the open web.
_MIMIC_DATA = ALL_TOOLS


# CSV column order for --csv. Module-level so any other writer can share it rather than
# maintaining a second copy that drifts.
BASE_COLS = [
    "label",
    "job",
    "n",
    "n_dirs",
    "missing_reward",
    "missing_gt",
    "no_report",
    "acc",
    "clean",
    "n_clean",
    "trap_adj",
    "raw_rp",
    "trap_gap",
    "correct_but_trapped",
    "macro_f1",
    "parse",
    "proc",
    "conf_ok",
    "conf_no",
    "ece",
    "n_conf",
    "cost_task",
    "cost_total",
    "turns",
    "toolcalls",
    "datacalls",
    "worktok",
    "out_tok",
    "mcp_timeouts",
    # APPENDED, never inserted: a mid-list insertion shifts every downstream column by one.
    "mcp_call_to_trials",
    "mcp_call_to_events",
    # Directional abstention (Section 5.1). The paper subscripts each denominator by the REFERENCE
    # label set, not the predicted one, so each rate's denominator is the opposite direction from
    # its own name: N_A = n_defer = reference-ABSTENTION runs = the over-COMMITMENT denominator;
    # N_D = n_defn = reference-DEFINITIVE runs = the over-ABSTENTION denominator.
    # APPENDED, never inserted.
    "over_com",
    "over_abs",
    "over_com_n",
    "over_abs_n",
    "n_defer",
    "n_defn",
    "n_unparsed_abst",
]


# Tool-name prefixes are OWNED by _template/tests/evidence.py (the hardlink constraint runs one
# way: it may not import eval/, but eval/ may import it). Keep ONE copy: a second list that
# drifts makes systems report zero data-access calls with no error.
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


_MCP_PREFIXES = _mcp_prefixes()


def _is_data_tool(name: str) -> bool:
    n = (name or "").strip()
    for pref in _MCP_PREFIXES:
        if n.startswith(pref):
            n = n[len(pref) :]
            break
    return n in _MIMIC_DATA


def _phase_seconds(phase: dict | None) -> float | None:
    """Duration in seconds from a Harbor {started_at, finished_at} ISO-8601 pair, or None."""
    if not isinstance(phase, dict):
        return None
    a, b = phase.get("started_at"), phase.get("finished_at")
    if not (a and b):
        return None
    try:
        t0 = datetime.fromisoformat(str(a).replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(str(b).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t1 - t0).total_seconds()


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float | None:
    """Expected Calibration Error over (confidence∈[0,1], correct∈{0,1}) pairs, 10 equal bins."""
    if not pairs:
        return None
    n = len(pairs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        grp = [(c, y) for c, y in pairs if (lo < c <= hi) or (b == 0 and c <= lo)]
        if not grp:
            continue
        conf = sum(c for c, _ in grp) / len(grp)
        acc = sum(y for _, y in grp) / len(grp)
        total += abs(conf - acc) * len(grp) / n
    return total


def amb_no_rule_scenarios() -> set[str]:
    """scenario_id (short form, matching rubrics.json) whose spec defines NO medical-ambiguity
    rule yet which carries >=1 gold AMB label in the cohort.

    Returns an empty set when the cohort bundle isn't configured, rather than raising: this
    only suppresses an ungradeable-cell warning in the report, so a caller scoring runs without
    the bundle to hand should still get their table.
    """
    out: set[str] = set()
    try:
        amb_scenarios = {r["scenario_id"] for r in load_full() if r["expected_label"] == AMB}
    except RuntimeError:
        return out
    for cdir in sorted(glob.glob(os.path.join(SCENARIOS_DIR, "scenario-*/"))):
        toml_p = os.path.join(cdir, "scenario.toml")
        if not os.path.exists(toml_p):
            continue
        try:
            spec = tomllib.load(open(toml_p, "rb"))
        except Exception:
            continue
        if spec.get("indeterminate_medical_ambiguity_criteria"):  # non-empty rule -> gradeable
            continue
        if spec.get("id") in amb_scenarios:
            # rubrics.json uses the short scenario_id (dir name minus the "scenario-NN-" prefix)
            name = os.path.basename(cdir.rstrip("/"))
            parts = name.split("-", 2)
            out.add(parts[2] if len(parts) == 3 and parts[1].isdigit() else name)
    return out


def row_label(job_dir: str) -> str:
    """harness × model, from a trial config.json (falls back to the job dir name)."""
    for cfg in glob.glob(os.path.join(job_dir, "*/config.json")):
        try:
            a = (json.load(open(cfg)) or {}).get("agent") or {}
        except Exception:
            continue
        raw_name = a.get("name") or ""
        if raw_name and ":" not in raw_name and "." not in raw_name:
            ident = raw_name  # built-in agent: claude-code / gemini-cli / codex / opencode
        elif raw_name or a.get("import_path"):
            # A custom agent ("clinicare_core.agents.codex_mcp_fix:CodexMcpFixed") arrives in EITHER field;
            # display the harness it wraps so rows read as harness × model.
            spec = raw_name or a["import_path"]
            mod = spec.split(":")[0].split(".")[-1]
            ident = {
                "codex_mcp_fix": "codex",
                "gemini_no_web": "gemini-cli",
                "opencode_local_xdg": "opencode",
            }.get(mod, mod.replace("_mcp_fix", "").replace("_fix", "").replace("_no_web", ""))
        else:
            ident = "?"
        model = (a.get("model_name") or "?").split("/")[-1]
        return f"{ident} × {model}"
    return os.path.basename(job_dir.rstrip("/"))


def job_task_dir(job_dir: str) -> str | None:
    """The task build this job ran, from its config (`datasets[0].path`, repo-root-relative).

    Authoritative for recovering the gold of a no-report trial: several `tasks*` builds coexist and
    their gold can differ, so the build must be pinned to the one that actually ran (see load_trial).
    """
    cfg = os.path.join(job_dir, "config.json")
    if not os.path.exists(cfg):
        return None
    try:
        ds = (json.load(open(cfg)) or {}).get("datasets") or []
        return (ds[0] or {}).get("path") if ds else None
    except Exception:
        return None


# --- process-axis regrade overlay --------------------------------------------------
# When CLINICARE_REGRADE_ROOT is set, the PROCESS axis alone -- process_pct and the
# sprung-trap flag -- is read from a regrade tree that mirrors <job>/<trial> flat:
#     $CLINICARE_REGRADE_ROOT/<job>/<trial>/{reward,grades_detail}.json
# The VERDICT axis is never overlaid. outcome_score, agent_label, confidence and every
# resource column keep coming from the source trial, so the overlay is structurally
# incapable of moving an accuracy or macro-F1 number -- which is the property that
# makes it safe to apply to tables whose verdict columns are already published.
# A trial with no overlay dir falls back to its source grades silently; that is
# correct rather than lossy, because a regrade only emits for trials whose
# frozen trace was gradeable.
REGRADE_ROOT = os.environ.get("CLINICARE_REGRADE_ROOT") or None
REGRADE_HITS = {"proc": 0, "trap": 0, "miss": 0}


def _sprung(process_graded: list) -> int:
    """1 if any MUST-NOT (negative-weight) criterion was graded "met"."""
    return int(
        any(
            str(c.get("weight", "")).strip().startswith("-")
            and str(c.get("grade", "")).strip().lower() == "met"
            for c in process_graded
        )
    )


def _regrade_dir(d: str) -> str | None:
    """Map a source trial dir to its overlay dir, or None if there is no overlay."""
    if not REGRADE_ROOT:
        return None
    parts = os.path.normpath(os.path.abspath(d)).split(os.sep)
    if len(parts) < 2:
        return None
    cand = os.path.join(REGRADE_ROOT, parts[-2], parts[-1])
    if os.path.isdir(cand):
        return cand
    REGRADE_HITS["miss"] += 1
    return None


def load_trial(d: str, task_dir: str | None = None) -> dict | None:
    rp = os.path.join(d, "verifier/reward.json")
    if not os.path.exists(rp):
        return None
    try:
        rj = json.load(open(rp))
    except Exception:
        return None
    if verifier_failed(rj):
        # The verifier failed before scoring, and its 0.0 is not a verdict: count the trial with the
        # infrastructure failures (missing_reward), not as a wrong answer. See clinicare_core.rewards.
        return None
    t: dict = {
        "score": int(
            (
                rj.get("outcome_score")
                if rj.get("outcome_score") is not None
                else rj.get("score") or 0
            )
            == 1
        ),
        "parsed": int(bool(rj.get("outcome_label_parsed"))),
        "proc": rj.get("process_pct") if not rj.get("process_judge_failed") else None,
        "gt": None,
        "scenario": None,
        "pred": None,
        "conf": None,
        "report_present": rj.get("report_present"),
        "turns": None,
        "toolcalls": None,
        "datacalls": None,
        "worktok": None,
        "outtok": None,
        "cost": None,
        "mcp_timeout": 0,
        "mcp_call_timeout": 0,
        "trapped": None,
        # consumed by eval/report/aggregate.py (separate PR); not CSV columns here, so
        # master's APPEND-only BASE_COLS ordering is untouched.
        "wall_s": None,
        "source": None,
    }
    # Process axis only; see _regrade_dir. Nothing below this touches t["score"].
    rgd = _regrade_dir(d)
    if rgd:
        rgp = os.path.join(rgd, "reward.json")
        if os.path.exists(rgp):
            try:
                rgj = json.load(open(rgp)) or {}
                if rgj.get("process_possible"):
                    t["proc"] = (
                        rgj.get("process_pct") if not rgj.get("process_judge_failed") else None
                    )
                    REGRADE_HITS["proc"] += 1
            except Exception:
                pass
    # This branch's in-trial verifier is DETERMINISTIC-ONLY, so reward.json carries no
    # `process_pct` at all — eval/metrics/process.py writes it to verifier/process.json instead.
    # Checked AFTER the regrade overlay above, so an explicit CLINICARE_REGRADE_ROOT still wins.
    if t["proc"] is None:
        pj = os.path.join(d, "verifier/process.json")
        if os.path.exists(pj):
            try:
                pd_ = json.load(open(pj)) or {}
                if not pd_.get("error"):
                    t["proc"] = (pd_.get("score") or {}).get("pct")
                    graded = pd_.get("graded") or []
                    if graded and t.get("trapped") is None:
                        t["trapped"] = _sprung(graded)
            except Exception:
                pass
    rb = os.path.join(d, "verifier/rubrics.json")
    if os.path.exists(rb):
        try:
            r = json.load(open(rb))
            t["gt"] = r.get("ground_truth_label")
            # `case_id`: pre-rename spelling, still in bundles captured under jobs/.
            t["scenario"] = r.get("scenario_id") or r.get("case_id")
        except Exception:
            pass
    # No-report trials early-exit before test.sh copies rubrics.json into the trial dir, so gt is
    # absent there. The gold is a property of the TASK, not the trial → recover it from the source
    # task's tests/rubrics.json (task-id = trial-dir name before "__"), so these still count per-class.
    # `task_dir` MUST be the build this job actually ran (job config.json: datasets[0].path).
    # Globbing `tasks*/` instead would pick an arbitrary filesystem-ordered build: 10 builds exist and
    # for 7 task ids the first hit was a stale variant whose gold DIFFERS, silently corrupting the
    # per-class supports (and hence recall/F1/macro-F1) of any system with a no-report trial.
    if t["gt"] is None and task_dir:
        task_id = os.path.basename(d.rstrip("/")).rsplit("__", 1)[0]
        src = os.path.join(BASE, task_dir, task_id, "tests/rubrics.json")
        if os.path.exists(src):
            try:
                r = json.load(open(src))
                t["gt"] = r.get("ground_truth_label")
                t["scenario"] = r.get("scenario_id") or r.get("case_id")
            except Exception:
                pass
    # Predicted label (for confusion / F1): grades_detail.json:agent_label.
    gd = os.path.join(d, "verifier/grades_detail.json")
    if os.path.exists(gd):
        try:
            gj = json.load(open(gd)) or {}
            t["pred"] = gj.get("agent_label")
            # Trap = a MUST-NOT (negative-weight) criterion the judge graded "met" — a sprung shortcut.
            pg = gj.get("process_graded") or []
            if pg:
                t["trapped"] = _sprung(pg)
        except Exception:
            pass
    # Overlay the trap flag from the regrade, but never t["pred"] -- the verdict stays
    # sourced from the original grade so the overlay cannot reorder the leaderboard.
    if rgd:
        rgg = os.path.join(rgd, "grades_detail.json")
        if os.path.exists(rgg):
            try:
                pg = (json.load(open(rgg)) or {}).get("process_graded") or []
                if pg:
                    t["trapped"] = _sprung(pg)
                    REGRADE_HITS["trap"] += 1
            except Exception:
                pass
    # Confidence (%) for calibration: the report's "## Confidence: NN%" line.
    rpt = os.path.join(d, "verifier/report.md")
    if os.path.exists(rpt):
        try:
            m = _CONF_RE.search(open(rpt, errors="ignore").read())
            if m:
                t["conf"] = float(m.group(1))
        except Exception:
            pass
    # Cost + tokens: result.json:agent_result is populated for ALL harnesses (gemini-cli's
    # trajectory.json omits total_cost_usd, so trajectory is only a fallback here).
    rr = os.path.join(d, "result.json")
    if os.path.exists(rr):
        try:
            rj_res = json.load(open(rr)) or {}
            ar = rj_res.get("agent_result") or {}
            t["cost"] = ar.get("cost_usd")
            t["source"] = rj_res.get("source")
            # AGENT-phase wall clock (excludes env/agent setup + verification) — the comparable one.
            t["wall_s"] = _phase_seconds(rj_res.get("agent_execution"))
            p, c, o = (
                (ar.get("n_input_tokens") or 0),
                (ar.get("n_cache_tokens") or 0),
                (ar.get("n_output_tokens") or 0),
            )
            if p or o:
                t["worktok"] = (p - c) + o
                t["outtok"] = (
                    o  # completion only — cache-independent, the fair cross-harness compute metric
                )
        except Exception:
            pass
    # Turns + tool-calls: trajectory.json (ATIF). final_metrics.total_steps = turns.
    tj = os.path.join(d, "agent/trajectory.json")
    if os.path.exists(tj):
        try:
            # Read ONCE as text, count the MCP per-call timeout marker, then parse. Re-opening the
            # file, or additionally reading the harness transcript, puts gigabytes of extra reads on
            # the canonical scoring path; on a busy host that walk can push `docker info` past
            # harbor's 10 s preflight abort and kill live trials.
            #
            # COVERAGE IS JOB-DEPENDENT -- do not quote a single percentage. The trajectory-only
            # scan catches most affected trials on some job dirs and well under half on others. So
            # this free scan is a genuine FLOOR and nothing more; the transcript scan below is
            # opt-in rather than paid for on every rescore.
            _raw = open(tj, errors="ignore").read()
            t["mcp_call_timeout"] += _raw.count("-32001")
            j = json.loads(_raw)
            fm = j.get("final_metrics") or {}
            t["turns"] = fm.get("total_steps")
            if t["cost"] is None:
                t["cost"] = fm.get("total_cost_usd")
            if t["worktok"] is None:
                p, c, o = (
                    (fm.get("total_prompt_tokens") or 0),
                    (fm.get("total_cached_tokens") or 0),
                    (fm.get("total_completion_tokens") or 0),
                )
                if p or o:
                    t["worktok"] = (p - c) + o
                    t["outtok"] = o
            tc = 0
            dc = 0
            for s in j.get("steps") or []:
                if not isinstance(s, dict):
                    continue
                names: list[str] = [
                    x.get("function_name")
                    or x.get("name")
                    or (x.get("function") or {}).get("name")
                    or ""
                    for x in (s.get("tool_calls") or [])
                ]
                if not any(names):  # fall back to message-level (harnesses that store there)
                    m = s.get("message") if isinstance(s.get("message"), dict) else {}
                    names = [
                        (x.get("function") or {}).get("name") or x.get("name") or ""
                        for x in (m.get("tool_calls") or [])
                    ]
                    cont = m.get("content")
                    if isinstance(cont, list):
                        names += [
                            b.get("name") or ""
                            for b in cont
                            if isinstance(b, dict) and b.get("type") == "tool_use"
                        ]
                names = [n for n in names if n]
                tc += len(names)
                dc += sum(1 for n in names if _is_data_tool(n))
            t["toolcalls"] = tc
            t["datacalls"] = dc
        except Exception:
            pass
    # Prefer verifier/evidence.jsonl for the data-call count: evidence.py extracted it from the
    # native trajectory and handles every harness, including codex "code mode" where the only
    # tool_call is `exec` and the real MIMIC calls live inside the script (counting tool_calls
    # there yields a silent 0 rather than an error).
    ev = os.path.join(d, "verifier/evidence.jsonl")
    if os.path.exists(ev):
        try:
            n_data = 0
            with open(ev, errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if (json.loads(line) or {}).get("kind") in ("mimic", "policy"):
                        n_data += 1
            t["datacalls"] = n_data
        except Exception:
            pass
    cx = os.path.join(d, "agent/codex.txt")
    if os.path.exists(cx):
        try:
            t["mcp_timeout"] = int("timed out handshaking" in open(cx, errors="ignore").read())
        except Exception:
            pass
    # `mcp_timeout` above is STRUCTURALLY BLIND outside codex, in two independent ways: it reads only
    # `agent/codex.txt`, which no other harness writes, and it looks only for the STARTUP handshake
    # string. On any other harness it therefore reports 0 -- meaning "unmeasurable", NOT "measured
    # zero" -- even where trials lost individual tool calls to `MCP error -32001: Request timed out`
    # mid-run. A harness defect then reads as model weakness, because the instrument that should
    # catch it cannot fire on the affected harness.
    #
    # This second field is harness-agnostic and counts PER-CALL timeouts on trials that SURVIVED,
    # which is the case the first field cannot see. Added rather than redefining `mcp_timeout`, so no
    # existing number changes meaning. The FREE part is counted above, off the trajectory.json read
    # this function already performs -- do not re-open that file here.
    #
    # OPT-IN, because it is not free: scanning harness transcripts recovers a small tail of
    # additional affected trials but costs a full extra multi-MB read per trial, and a heavy walk
    # on a busy host can push `docker info` past harbor's 10 s preflight abort and kill live
    # trials. Default OFF makes the column a documented FLOOR rather than a silent undercount --
    # the same distinction that makes a bare 0 misleading when it means "unmeasurable".
    #
    # The transcript sets only the TRIAL FLAG, never the event count, and is skipped once the trajectory
    # has already flagged the trial. Do NOT add the two: some harness transcripts EMBED the
    # trajectory JSON, sometimes several times over, so the same events are counted repeatedly and
    # neither `+` nor `max` is a floor -- only the canonical count is. A transcript-only trial
    # scores 1: it proves "at least one", not how many.
    if os.environ.get("MCP_SCAN_TRANSCRIPTS") == "1" and not t["mcp_call_timeout"]:
        for leaf in ("opencode.txt", "codex.txt", "claude-code.txt", "qwen-code.txt", "gemini.txt"):
            p = os.path.join(d, "agent", leaf)
            if not os.path.exists(p):
                continue
            try:
                if "-32001" in open(p, errors="ignore").read():
                    t["mcp_call_timeout"] = 1
                    break
            except Exception:
                pass
    return t


def score_job(job_dir: str, norule: set[str]) -> dict:
    dirs = glob.glob(os.path.join(job_dir, "*/"))
    task_dir = job_task_dir(job_dir)
    if task_dir is None:
        print(
            f"WARNING {os.path.basename(job_dir.rstrip('/'))}: no datasets[0].path in config.json -- "
            "gold for no-report trials cannot be recovered; their per-class support is dropped.",
            file=sys.stderr,
        )
    trials = [t for t in (load_trial(d, task_dir) for d in dirs) if t]
    return score_trials(
        row_label(job_dir), os.path.basename(job_dir.rstrip("/")), trials, len(dirs), norule
    )


def score_trials(label: str, job: str, trials: list[dict], n_dirs: int, norule: set[str]) -> dict:
    """The metric math, over an already-loaded trial list.

    Factored out of score_job so that a caller holding a MERGED trial list -- e.g. one system's cohort
    union across two job dirs, because the docker sweep owns tasks 1-500 and a modal sweep owns
    501-750 -- gets numbers from this exact code path instead of a second implementation. Every
    denominator subtlety lives here (trap_adj over report-present only, trap_gap vs raw_rp, the
    AMB-no-rule exclusion), and those are precisely the things a re-implementation gets wrong.

    `trials` must be deduplicated by task id already; this function cannot detect a task counted
    twice. `n_dirs` is the number of candidate trial dirs the list was drawn from, so that
    missing_reward (the infrastructure failure) stays distinguishable from a no-report trial (a
    legitimate scored result).
    """
    n = len(trials)
    missing_reward = n_dirs - n  # no usable reward: none written, or the verifier failed first
    missing_gt = sum(1 for t in trials if not t["gt"])  # gold still unrecoverable (rare)
    no_report = sum(1 for t in trials if t.get("report_present") == 0)  # agent wrote no deliverable
    med = lambda xs: st.median(xs) if xs else 0
    per = defaultdict(lambda: [0, 0])
    for t in trials:
        if t["gt"]:
            per[t["gt"]][1] += 1
            per[t["gt"]][0] += t["score"]
    clean = [t for t in trials if not (t["scenario"] in norule and t["gt"] == AMB)]
    procs = [t["proc"] for t in trials if t["proc"] is not None]
    costs = [t["cost"] for t in trials if t["cost"] is not None]

    # Per-class precision/recall/F1 from the confusion (predicted × gold). An unparsed/none
    # prediction counts as a false-negative for its gold class (no predicted class to credit).
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for t in trials:
        gt, pred = t["gt"], t["pred"]
        if not gt:
            continue
        if pred in CLASSES:
            if pred == gt:
                tp[gt] += 1
            else:
                fp[pred] += 1
                fn[gt] += 1
        else:
            fn[gt] += 1
    # Directional abstention errors (Section 5.1), denominators emitted alongside the rates:
    #   over-commitment = |gt in A, pred in D| / |gt in A|   D = {YES, NO}, A = {lack, amb}
    #   over-abstention = |gt in D, pred in A| / |gt in D|
    # An unparsed prediction is not a direction: excluded from the numerators, kept in the
    # denominator support, counted as `n_unparsed_abst`. Mirrors agg_full_cohort.py, which produced
    # the published cells (verified: 42/130 and 90/620 reproduce GPT-5.5's 32.3 / 14.5).
    DEFINITIVE, DEFERRAL = {"YES", "NO"}, {LACK, AMB}
    n_defer = sum(1 for t in trials if t["gt"] in DEFERRAL)  # N_A: reference-abstention support
    n_defn = sum(1 for t in trials if t["gt"] in DEFINITIVE)  # N_D: reference-definitive support
    over_com_n = sum(1 for t in trials if t["gt"] in DEFERRAL and t["pred"] in DEFINITIVE)
    over_abs_n = sum(1 for t in trials if t["gt"] in DEFINITIVE and t["pred"] in DEFERRAL)
    n_unparsed_abst = sum(
        1 for t in trials if t["gt"] in (DEFERRAL | DEFINITIVE) and t["pred"] not in CLASSES
    )

    f1, prec = {}, {}
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        prec[c] = p
        f1[c] = 2 * p * r / (p + r) if (p + r) else 0.0
    macro_f1 = sum(f1.values()) / len(CLASSES)

    # Calibration: confidence when correct vs wrong, and ECE (conf∈[0,1] vs correctness).
    conf_ok = [t["conf"] for t in trials if t["conf"] is not None and t["score"] == 1]
    conf_no = [t["conf"] for t in trials if t["conf"] is not None and t["score"] == 0]
    ece_val = ece([(t["conf"] / 100.0, t["score"]) for t in trials if t["conf"] is not None])

    # Trap-adjusted accuracy (adversarial robustness), over report-present trials — a no-report trial
    # has no process grades. A trial counts correct only if the label is right AND no MUST-NOT trap was
    # sprung. `correct_but_trapped` = share of correct verdicts reached via a sprung shortcut.
    rp = [t for t in trials if t.get("report_present") != 0]
    corr = [t for t in rp if t["score"] == 1]
    trap_adj = (
        100 * sum(1 for t in rp if t["score"] == 1 and not t.get("trapped")) / len(rp)
        if rp
        else None
    )
    raw_rp = 100 * len(corr) / len(rp) if rp else None
    correct_but_trapped = (
        100 * sum(1 for t in corr if t.get("trapped")) / len(corr) if corr else None
    )
    return {
        "label": label,
        "job": job,
        "n": n,
        "n_dirs": n_dirs,
        "missing_reward": missing_reward,
        "missing_gt": missing_gt,
        "no_report": no_report,
        "acc": 100 * sum(t["score"] for t in trials) / n if n else 0,
        "clean": 100 * sum(t["score"] for t in clean) / len(clean) if clean else 0,
        "n_clean": len(clean),
        # trap_gap is emitted rather than left to the caller because the obvious subtraction is
        # WRONG: `acc` is over ALL trials (a no-report trial counts incorrect) while `trap_adj` is
        # over report-present trials only, so `acc - trap_adj` conflates non-delivery with sprung
        # traps. On one sweep that made the weakest system look like the *least* trap-prone (a 3.5-pt
        # apparent gap that was really 7.8) and put a wrong range in a paper draft. `raw_rp` is the
        # like-for-like numerator; this is the only difference that means "cost of sprung traps".
        "trap_adj": trap_adj,
        "raw_rp": raw_rp,
        "correct_but_trapped": correct_but_trapped,
        "trap_gap": (raw_rp - trap_adj) if (raw_rp is not None and trap_adj is not None) else None,
        "parse": 100 * sum(t["parsed"] for t in trials) / n if n else 0,
        "proc": st.mean(procs) if procs else None,
        "per": {c: (per[c][0], per[c][1]) for c in CLASSES},
        "cost_task": st.mean(costs) if costs else None,
        "cost_total": sum(costs) if costs else None,
        "turns": med([t["turns"] for t in trials if t["turns"] is not None]),
        "toolcalls": med([t["toolcalls"] for t in trials if t["toolcalls"] is not None]),
        "datacalls": med([t["datacalls"] for t in trials if t["datacalls"] is not None]),
        "worktok": med([t["worktok"] for t in trials if t["worktok"] is not None]),
        "out_tok": med([t["outtok"] for t in trials if t["outtok"] is not None]),
        "mcp_timeouts": sum(t["mcp_timeout"] for t in trials),
        # trials with >=1 per-call timeout, and the total event count. The TRIAL count is the one to
        # read: the event count is dominated by a few trials that retried a dozen times, and it is a
        # floor off the canonical trajectory only -- transcripts add trials, not events (see load_trial).
        "mcp_call_to_trials": sum(1 for t in trials if t["mcp_call_timeout"]),
        "mcp_call_to_events": sum(t["mcp_call_timeout"] for t in trials),
        "f1": f1,
        "prec": prec,
        "macro_f1": macro_f1,
        "conf_ok": st.mean(conf_ok) if conf_ok else None,
        "conf_no": st.mean(conf_no) if conf_no else None,
        "ece": ece_val,
        "n_conf": len(conf_ok) + len(conf_no),
        # None (not 0.0) on an empty support, so "none graded" never reads as "no errors".
        "over_com": (100.0 * over_com_n / n_defer) if n_defer else None,
        "over_abs": (100.0 * over_abs_n / n_defn) if n_defn else None,
        "over_com_n": over_com_n,
        "over_abs_n": over_abs_n,
        "n_defer": n_defer,
        "n_defn": n_defn,
        "n_unparsed_abst": n_unparsed_abst,
    }


def pct(cell: tuple[int, int]) -> str:
    c, tot = cell
    return f"{100 * c / tot:.0f}%" if tot else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description="CliniCare-Bench NxM scorer")
    ap.add_argument("jobs", nargs="+", help="job dirs or globs")
    ap.add_argument("--csv", help="also write full metrics to this CSV")
    args = ap.parse_args()

    job_dirs = [d for pat in args.jobs for d in glob.glob(pat) if os.path.isdir(d)]
    if not job_dirs:
        print("no job dirs matched", file=sys.stderr)
        return 1
    norule = amb_no_rule_scenarios()
    rows = [score_job(d, norule) for d in sorted(set(job_dirs))]

    print(
        "\n| harness × model | n | **acc** | **trap-adj** | **mF1** | proc% | parse | "
        + " | ".join("rec:" + SHORT[c] for c in CLASSES)
        + " | USD/task | USD-total | turns | tools | data | work-tok | mcp-to tr/ev |"
    )
    print("| " + " | ".join(["---"] * (7 + len(CLASSES) + 7)) + " |")
    for r in rows:
        proc = f"{r['proc']:.1f}" if r["proc"] is not None else "-"
        ta = f"{r['trap_adj']:.1f}" if r.get("trap_adj") is not None else "-"
        # plain USD numbers (no `$` glyph) — a `$` in the cells pairs as LaTeX math and mangles
        # the table in MathJax-enabled markdown viewers (GitHub, etc.). Header cols carry the unit.
        ct = f"{r['cost_task']:.2f}" if r["cost_task"] is not None else "-"
        tot = f"{r['cost_total']:.0f}" if r["cost_total"] is not None else "-"
        cls = " | ".join(pct(r["per"][c]) for c in CLASSES)
        print(
            f"| {r['label']} | {r['n']} | {r['acc']:.1f}% | {ta} | {r['macro_f1']:.3f} | "
            f"{proc} | {r['parse']:.0f}% | {cls} | {ct} | {tot} | {r['turns']:.0f} | {r['toolcalls']:.0f} | "
            f"{r['datacalls']:.0f} | {r['worktok'] / 1000:.0f}k | "
            f"{r['mcp_call_to_trials']}/{r['mcp_call_to_events']} |"
        )
    print(
        "\nrec:YES/NO/lack/amb = per-class **recall** (% of that gold class labeled correctly; overall "
        "accuracy is the **acc** column) · mF1 = macro-F1 over the 4 classes · trap-adj = correct label "
        "AND no MUST-NOT sprung · **mcp-to tr/ev** = trials with >=1 `MCP error -32001` per-call timeout "
        "/ occurrences. A documented **FLOOR**: nonzero means that system's scores are depressed by a "
        "harness defect, but 0 means 'none seen', NOT healthy (the transcript scan is opt-in via "
        "`MCP_SCAN_TRANSCRIPTS=1`; read the trial count, not the event count) · **data** = MIMIC tool "
        "calls, read from `verifier/evidence.jsonl`, which normalizes each harness's dialect (a bare "
        "`mimic_` prefix, code-mode `exec` scripts); without that file the count is within-harness at "
        "best, and code-mode counts are a LOWER bound (a loop body appears once in the script text) · "
        "**tools** (total) mixes data calls with harness overhead · turns are within-harness only · "
        "cost in USD at the configured rates, including the cache-read discount; a harness that reports "
        "no cost reads 0 · work-tok = prompt−cached+completion, NOT cross-harness comparable (a harness "
        "reporting cache=0 inflates it) — use output tokens for a cache-neutral compare"
    )

    # Calibration + per-class precision & F1 (recall is in the main table). The deep-dive per-class table.
    print(
        "\n| harness × model | conf✓ | conf✗ | gap | ECE | "
        + " | ".join(f"prec:{SHORT[c]}" for c in CLASSES)
        + " | "
        + " | ".join(f"F1·{SHORT[c]}" for c in CLASSES)
        + " |"
    )
    print("| " + " | ".join(["---"] * (5 + 2 * len(CLASSES))) + " |")
    for r in rows:
        co, cn = r["conf_ok"], r["conf_no"]
        gap = f"{co - cn:+.0f}" if (co is not None and cn is not None) else "-"
        cs = f"{co:.0f}%" if co is not None else "-"
        ws = f"{cn:.0f}%" if cn is not None else "-"
        es = f"{r['ece']:.3f}" if r["ece"] is not None else "-"
        pc = " | ".join(f"{r['prec'][c]:.2f}" for c in CLASSES)
        f1c = " | ".join(f"{r['f1'][c]:.2f}" for c in CLASSES)
        print(f"| {r['label']} | {cs} | {ws} | {gap} | {es} | {pc} | {f1c} |")
    print(
        "\nconf✓/✗ = mean stated confidence when correct/wrong (from report `## Confidence:`); a small gap + "
        "high ECE = overconfident · **prec:C** = per-class **precision** (when the model says class C, how "
        "often it's right — the false-alarm side of the recall in the main table) · **F1·C** = per-class F1 "
        "(precision+recall combined)."
    )

    # Surface trial-health issues instead of silently dropping them.
    warns = [r for r in rows if r["missing_reward"] or r["missing_gt"] or r["no_report"]]
    if warns:
        print("\n⚠️  trial health:")
        for r in warns:
            bits = []
            if r["no_report"]:
                bits.append(
                    f"{r['no_report']} wrote no report.md (no deliverable → scored wrong; "
                    "gold recovered from source, so counted in per-class)"
                )
            if r["missing_reward"]:
                bits.append(
                    f"{r['missing_reward']} unscored: no reward.json, or the verifier failed "
                    "before scoring (dropped from n)"
                )
            if r["missing_gt"]:
                bits.append(f"{r['missing_gt']} gold unrecoverable (excluded from per-class/clean)")
            print(f"   {r['label']}: {r['n']}/{r['n_dirs']} scored — " + "; ".join(bits))

    if args.csv:
        base_cols = list(BASE_COLS)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                base_cols
                + [f"recall_{SHORT[c]}" for c in CLASSES]
                + [f"prec_{SHORT[c]}" for c in CLASSES]
                + [f"f1_{SHORT[c]}" for c in CLASSES]
            )
            for r in rows:
                rec = [
                    round(100 * r["per"][c][0] / r["per"][c][1], 1) if r["per"][c][1] else ""
                    for c in CLASSES
                ]
                prc = [round(100 * r["prec"][c], 1) for c in CLASSES]
                f1c = [round(r["f1"][c], 3) for c in CLASSES]
                w.writerow([r[k] for k in base_cols] + rec + prc + f1c)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
