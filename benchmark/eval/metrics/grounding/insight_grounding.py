"""insight_grounding — record-identifier precision (P_record in the paper).

Are the report's cited record IDs grounded in what the agent retrieved? Adapted from DRBench's
`factuality` (claim ↔ cited-source LLM judge) to CliniCARE-Bench. The
healthcare adaptation is trace-only: the "source" is the exact MIMIC rows the agent saw
(the trajectory's observation pool), so a claim is grounded iff it traces back to data the
agent actually retrieved — no live MIMIC, no gold set.

This file is **Tier 1 (T1): the deterministic guardrail**. It checks the *hard atoms* of a
report — MIMIC **record IDs** (subject / hadm / stay / poe / note / order ids) — for
verbatim presence in the retrieved evidence:

  grounding = |cited IDs seen in evidence| / |cited IDs|     (per trial)

An ID is either something the agent read or something it fabricated — you cannot *compute*
an ID — so an ID absent from every tool result is a fabrication. That makes ID-grounding a
sound **fabrication floor**: 1.00 means "no invented record IDs".

Deliberately **only record IDs**, because the other atom types are not deterministically
checkable:
  - **timestamps** — reports legitimately state *computed* values (e.g. a 7-day window
    boundary = CT time − 7 days) that are never a DB row; a verbatim check wrongly flags
    correct arithmetic as ungrounded. You compute timestamps; you don't compute IDs.
  - **bare decimals** (lab values like `0.9`) — substring-match by chance, so a verbatim
    check is not discriminative.
Both belong to **T2 (the LLM judge)**, which checks numeric / free-text claims against the
cited evidence in context. T2 is where the discriminative grounding signal lives; T1 is the
cheap deterministic backstop under it.

Note on interpretation: on capable models T1 tends to read ~1.00 (they copy IDs faithfully),
so it *separates* models poorly while still *catching* fabrication. Treat it as a floor /
regression guard, not a leaderboard axis — the model comparison is T2's job.

  cd <repo root>
  python benchmark/eval/metrics/grounding/insight_grounding.py \
      --traces benchmark/eval/traces [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from clinicare_core.paths import bootstrap

HERE = Path(__file__).resolve().parent

bootstrap(__file__, "eval")
from core.trajectory import load_trajectory, subagent_observations  # noqa: E402

# ── atom extraction ───────────────────────────────────────────────────────────────────────
# Dates/timestamps first, so they're stripped before ID matching. Covers full timestamps,
# YYYY-MM-DD dates, AND bare YYYY-MM timeline labels (reports use "2186-12" as month markers)
# — all of which would otherwise parse as hyphenated IDs (4-digit year + "-12"). The \b
# anchors keep this from biting into a real ID's digits (10001217-107 has no internal
# boundary). Timestamps aren't scored themselves (see module doc).
_TS = re.compile(r"\b\d{4}-\d{2}(?:-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?)?\b")
# Hyphenated record IDs: 10001217-107 (poe_id), 10001217-RR-14 (note_id), 10001217-DS-21.
# Prefix is the 8-digit subject_id, so require ≥6 digits — a 4-digit year can't anchor an ID.
_ID_HYPHEN = re.compile(r"\b\d{6,}-(?:[A-Z]{1,4}-)?\d+\b")
# Long integer IDs: subject_id / hadm_id / stay_id are 8 digits (≥6 to be safe).
_ID_INT = re.compile(r"\b\d{6,}\b")


def _extract_ids(text: str, drop_compound_prefixes: bool) -> set[str]:
    """Record-ID atoms in `text`. Dates/timestamps are stripped first so they can't be
    mis-read as IDs. When `drop_compound_prefixes` is set, hyphenated IDs are removed before
    the int pass so the 8-digit prefix of a compound (10001217 in 10001217-107) isn't counted
    as its own atom."""
    text = _TS.sub(" ", text)
    ids = {m.group() for m in _ID_HYPHEN.finditer(text)}
    if drop_compound_prefixes:
        text = _ID_HYPHEN.sub(" ", text)
    ids |= {m.group() for m in _ID_INT.finditer(text)}
    return ids


def cited_ids(report_text: str) -> set[str]:
    """Distinct record-ID atoms the *report* cites (numerator). Compound prefixes are dropped
    so a poe_id like 10001217-107 counts once, not also as a bare subject_id the report never
    actually cited on its own."""
    return _extract_ids(report_text, drop_compound_prefixes=True)


def pool_ids(pool: str) -> set[str]:
    """Grounding targets extractable from the *evidence* (membership set). Unlike the report
    side, compound prefixes are KEPT: a subject_id may appear in the rows only inside compound
    IDs (10001217-6, 10001217-RR-14), yet a report legitimately cites it bare as 10001217, so
    that prefix must be a valid target. Word-boundary extraction still means a fabricated
    numeric *prefix* (100012 of a real 10001234) is not a token here → correctly ungrounded."""
    return _extract_ids(pool, drop_compound_prefixes=False)


def evidence_observations(trial_dir: Path) -> tuple[list, bool, bool]:
    """The evidence the agent saw, as a list of per-call Observations (step_id, tool, args, text),
    plus (used_subagent, subagent_recovered). Same content as :func:`evidence_pool`, kept
    per call so a claim can be judged against the specific records it cites: sub-agent
    tool-call args become Observations with tool ``_subagent_args`` and the args as text."""
    traj = load_trajectory(trial_dir / "trajectory.json")
    obs = list(traj.observations)
    sub = subagent_observations(trial_dir) if traj.used_subagent else []
    for o in sub:
        if isinstance(o.args, dict):
            obs.append(
                type(o)(step_id=o.step_id, tool="_subagent_args", args=o.args,
                        text=" ".join(str(v) for v in o.args.values()))
            )
    return obs, traj.used_subagent, bool(sub)


def evidence_pool(trial_dir: Path) -> tuple[str, bool, bool]:
    """The text the agent actually saw, to check citations against. Returns
    (pool_text, used_subagent, subagent_recovered).

    Pool = every MIMIC observation's returned rows + the parent-visible sub-agent summaries
    (both from trajectory.json) + any recovered sub-agent tool-call args (subagents/*.jsonl).
    Sub-agent *result rows* aren't captured (the session JSONL keeps only tool_use blocks),
    but an ID the sub-agent passed as an argument came from data it had seen, so folding args
    in keeps the floor from false-flagging IDs that only a sub-agent surfaced."""
    traj = load_trajectory(trial_dir / "trajectory.json")
    parts = [traj.evidence_text]
    sub = subagent_observations(trial_dir) if traj.used_subagent else []
    for o in sub:
        if isinstance(o.args, dict):
            parts.append(" ".join(str(v) for v in o.args.values()))
    return "\n".join(parts), traj.used_subagent, bool(sub)


def score(report_text: str, pool: str) -> dict:
    ids = cited_ids(report_text)
    # Compare against IDs extracted from the pool as a SET — not `x in pool` substring search,
    # which would count a fabricated ID that is a numeric prefix of a real pool ID (e.g.
    # "100012" inside "10001234") as grounded and defeat the fabrication floor.
    targets = pool_ids(pool)
    ungrounded = sorted(x for x in ids if x not in targets)
    grounding = 1 - len(ungrounded) / len(ids) if ids else None
    return {
        "grounding": grounding,
        "n_ids": len(ids),
        "ungrounded": ungrounded,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=HERE.parents[1] / "traces")
    ap.add_argument("--verbose", action="store_true", help="list every cited ID per trial")
    args = ap.parse_args()

    trials = sorted(args.traces.glob("*/*/*/trajectory.json"))
    if not trials:
        sys.exit(f"no trajectories under {args.traces}")

    rows, rates = [], []
    for tj in trials:
        # A trace trial dir holds `agent/trajectory.json` AND `verifier/report.md` (see
        # eval/README.md), so the two live in SIBLING dirs — the report is not beside the
        # trajectory. evidence_pool() wants the agent dir; the report is under verifier/.
        agent_dir = tj.parent
        trial_dir = tj.parents[1]
        report = trial_dir / "verifier" / "report.md"
        if not report.exists():
            print(f"  ! {trial_dir.name}: no verifier/report.md", file=sys.stderr)
            continue
        pool, used_sub, recovered = evidence_pool(agent_dir)
        s = score(report.read_text(), pool)
        s["all_ids"] = sorted(cited_ids(report.read_text()))
        traj = load_trajectory(tj)
        s["used_subagent"] = used_sub
        # A fabrication flagged on a sub-agent trial whose sessions weren't fetched is
        # low-confidence: the ID may live in unretrieved sub-agent result rows.
        s["low_confidence"] = bool(s["ungrounded"]) and used_sub and not recovered
        rows.append((tj.parts[-4], tj.parts[-3], traj.agent, s))
        if s["grounding"] is not None:
            rates.append(s["grounding"])

    print(
        f"{'run':30s} {'scenario':18s} {'grounded':>8s} {'#ids':>5s}  fabricated (ungrounded) IDs"
    )
    for run, slug, agent, s in rows:
        g = f"{s['grounding']:.2f}" if s["grounding"] is not None else "  -"
        flag = " [low-confidence: subagent sessions not fetched]" if s["low_confidence"] else ""
        print(
            f"  {run[:28]:28s} {slug:18s} {g:>8s} {s['n_ids']:>5d}  {s['ungrounded'] or ''}{flag}"
        )
        if args.verbose:
            print(f"      cited ids: {s['all_ids']}")

    if rates:
        floor = sum(1 for r in rates if r == 1.0)
        print(f"\nmean grounding over {len(rates)} trials: {sum(rates)/len(rates):.3f}")
        print(f"trials with no fabricated IDs (grounding=1.00): {floor}/{len(rates)}")

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
