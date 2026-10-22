"""Post-hoc process-rubric scorer — the LLM process grade, run outside the trial.

Grades each trial's process rubric from the artifacts the run persisted: ``verifier/report.md``,
``agent/trajectory.json``, and the ``verifier/rubrics.json`` captured at build time. Because the
inputs are on disk, re-grading is repeatable and costs no re-sweep — an edited rubric is a
re-grade, not a new set of agent trajectories.

There is one definition of the grade, not two: the prompt and scoring live in
``eval/core/judge_core.py``, shared with the judge-agreement study in ``eval/reliability/``, and
judge_panel's ``_agent_work`` / ``_find_rubrics`` turn a trajectory into judge input exactly one
way.

    export JUDGE_BASE_URL=... JUDGE_API_KEY=...
    python benchmark/eval/metrics/process.py jobs/<job>             # grade
    python benchmark/eval/metrics/process.py jobs/<job> --dry-run   # coverage only, no LLM

Writes ``<trial>/verifier/process.json`` per trial (idempotent — re-runs skip existing files
unless ``--force``) and ``<job>/process_summary.json`` per job.

COVERAGE GATE: every trial dir is enumerated and classified, and the exit code is non-zero if any
trial failed for a reason that is OURS to fix (unreadable rubrics, judge error). Trials that are
legitimately unscorable — the agent wrote no report, or the scenario has no process rubric — are
reported separately and do NOT fail the run. A metric may not report a headline number without
its denominator, so ``--limit`` prints a loud PARTIAL banner and suppresses the gate.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clinicare_core import write_reward_atomic
from clinicare_core.paths import bootstrap

EVAL = Path(__file__).resolve().parents[1]  # benchmark/eval

bootstrap(__file__, "reliability", "eval")
import judge_panel as JP  # noqa: E402  (also gives us JP.J = the real judge module)
from core.judge_core import _build_reference, _grade  # noqa: E402

# Bumped when a change here alters the numbers, so a figure can be traced to the code that made
# it. Grading logic itself lives in
# judge.py — a change THERE is captured by judge_schema below, not this constant.
SCORER_VERSION = 1

# Classifications that mean "we could not score this and that is a problem".
FAILURE_REASONS = ("no_rubrics", "judge_error", "unverified")
# Classifications that mean "nothing to score here, correctly".
BENIGN_REASONS = ("no_report", "no_process_rubrics", "no_policy_rubrics", "policy_off")


def classify(trial: Path, rubrics_from: Path | None, require: str = "process_rubrics") -> dict:
    """Everything needed to grade one trial, or the reason it can't be graded.

    Deliberately returns a record for EVERY trial dir (rather than filtering, as
    judge_panel._collect_trials does) — the coverage gate needs the skips, not just the
    gradeable set.

    ``require`` is the rubrics.json key that must be non-empty for this trial to be gradeable
    (``process_rubrics`` here, ``policy_rubrics`` for the policy scorer); its absence yields the
    benign reason ``no_<key>``. Pass None to require nothing.
    """
    task_id = trial.name.rsplit("__", 1)[0]
    rec: dict = {"trial": trial.name, "task_id": task_id, "path": trial, "reason": None}

    report = trial / "verifier" / "report.md"
    if not report.is_file():
        # Two very different causes, and the gate must not merge them:
        #   * reward.json present  -> the verifier ran and the agent wrote no deliverable. The
        #     verdict is scored wrong by test.sh, which is correct. Benign.
        #   * reward.json absent   -> the verifier never ran (skipped by config, killed job).
        #     That is infrastructure, i.e. our fault, and must fail the gate.
        rec["reason"] = (
            "no_report" if (trial / "verifier" / "reward.json").is_file() else "unverified"
        )
        return rec

    rub, rub_src = JP._find_rubrics(trial, task_id, rubrics_from)
    if rub is None:
        rec["reason"] = "no_rubrics"
        return rec
    try:
        bundle = json.loads(Path(rub).read_text())
    except (OSError, ValueError) as exc:
        rec["reason"] = "no_rubrics"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec

    if require and not bundle.get(require):
        rec["reason"] = f"no_{require}"
        return rec

    rec.update(
        report=report,
        rub=Path(rub),
        rub_src=rub_src,
        bundle=bundle,
        traj=next(
            (
                trial / c
                for c in ("agent/trajectory.json", "agent/output.txt", "agent/stdout.txt")
                if (trial / c).is_file()
            ),
            None,
        ),
    )
    return rec


def grade_one(rec: dict, model: str, effort: str | None, base_url: str, api_key: str) -> dict:
    """Grade one trial and return the process.json payload (also on failure, with `error` set)."""
    bundle = rec["bundle"]
    report = JP.J._read_text(str(rec["report"]))
    trajectory = JP.J._read_trajectory(str(rec["traj"])) if rec["traj"] else ""
    out: dict = {
        "schema": "healthcare/process.v1",
        "scorer_version": SCORER_VERSION,
        # Which judge produced this, and what it read — enough to reproduce or invalidate the row.
        "judge_model": model + (f"@{effort}" if effort else ""),
        "task_id": rec["task_id"],
        # `case_id`: pre-rename spelling, still present in bundles captured under jobs/.
        "scenario_id": bundle.get("scenario_id") or bundle.get("case_id"),
        "ground_truth_label": (bundle.get("ground_truth_label") or "").upper(),
        "rubrics_source": rec["rub_src"],
        # Exact input identity, checked by cached_ok. report_chars/trajectory_chars stay as
        # human-readable context, but a same-length edit slips past them — the hash does not.
        "inputs_sha256": rec.get("inputs_sha256", ""),
        "judge_base_url": base_url,
        "report_chars": len(report),
        "trajectory_chars": len(trajectory),
        "trajectory_present": 1 if trajectory else 0,
    }
    try:
        res = _grade(
            bundle["process_rubrics"],
            bundle.get("prompt", ""),
            _build_reference(bundle),
            JP._agent_work(report, trajectory),
            model,
            base_url,
            api_key,
            reasoning_effort=effort,
        )
    except BaseException as exc:  # noqa: BLE001 — one bad trial must not kill the sweep
        out["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        return out
    out["graded"] = res["graded"]
    out["score"] = res["score"]
    if "chunk_info" in res:
        out["chunk_info"] = res["chunk_info"]
    return out


def amend_reward(trial: Path, payload: dict) -> bool:
    """Merge the process_* keys into the trial's reward.json (opt-in; it mutates an artifact
    Harbor wrote). Kept flat float|int, matching Harbor's reward contract."""
    rp = trial / "verifier" / "reward.json"
    try:
        reward = json.loads(rp.read_text())
    except (OSError, ValueError):
        return False
    sc = payload.get("score") or {}
    reward.update(
        {
            "process_earned": sc.get("earned", 0.0),
            "process_possible": sc.get("possible", 0.0),
            "process_pct": sc.get("pct", 0.0),
            "process_judge_failed": 1 if payload.get("error") else 0,
        }
    )
    write_reward_atomic(rp, reward)  # a run_case.py sharing this job may be reading it
    return True


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def inputs_sha(rec: dict, rubric_key: str = "process_rubrics") -> str:
    """Hash of everything a grading of this trial reads: the report, the trajectory, and the
    rubrics. Cheap — all 750 reports+trajectories of a full job hash in ~4 s — so it is affordable
    to compute for cached trials too, which is what makes input-aware validation possible.

    Shared with policy_support (pass rubric_key='policy_rubrics'), which classifies through the
    same `classify()` and therefore has the same record shape.
    """
    report = JP.J._read_text(str(rec["report"])) if rec.get("report") else ""
    traj = JP.J._read_trajectory(str(rec["traj"])) if rec.get("traj") else ""
    rubrics = json.dumps((rec.get("bundle") or {}).get(rubric_key) or [], sort_keys=True)
    return _sha(report, traj, rubrics)


def cached_ok(path: Path, expect: dict) -> bool:
    """True only if `path` holds a SUCCESSFUL prior grading OF THE SAME INPUTS.

    Two independent rejections:
      * an `error` payload (judge failure, bad model name, expired key) must be retried — otherwise
        a transient failure caches permanently and the coverage gate reports it forever;
      * a key mismatch. `judge_model` here already carries the reasoning effort (`model@effort`),
        so `--judge-model` and `--effort` both invalidate; `inputs_sha256` covers an edited report,
        a re-run trajectory, or changed rubrics.

    A payload written before the key has no `inputs_sha256` and is stale by construction, so it
    is re-graded once. That is the correct default: a cache whose inputs cannot be verified must
    not be trusted.
    """
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if payload.get("error"):
        return False
    return all(payload.get(k) == v for k, v in expect.items())


def _met(graded: list[dict]) -> dict[str, int]:
    return {
        g.get("id", ""): (1 if str(g.get("grade", "")).lower() in ("met", "pass") else 0)
        for g in graded
        if g.get("id")
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("jobs", nargs="+", help="job dirs or globs")
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", "openai/gpt-5.5"),
        help="'model' or 'model@effort' (effort → reasoning_effort), matching judge_panel",
    )
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument(
        "--force", action="store_true", help="re-grade trials that already have process.json"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="classify + report coverage without calling the judge (no LLM, no cost)",
    )
    ap.add_argument(
        "--amend-reward",
        action="store_true",
        help="also merge process_* keys back into each trial's reward.json (mutates it)",
    )
    ap.add_argument(
        "--rubrics-from",
        default=None,
        metavar="TASKS_DIR",
        help="grade against the CURRENT rubrics in this built-tasks dir instead of each trial's "
        "captured copy (e.g. benchmark/tasks_answer) — for re-grading with edited rubrics",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap trials per job (0=all); suppresses the coverage gate",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report coverage failures but exit 0 anyway",
    )
    args = ap.parse_args()

    job_dirs = sorted({d for pat in args.jobs for d in glob.glob(pat) if os.path.isdir(d)})
    if not job_dirs:
        print("no job dirs matched", file=sys.stderr)
        return 1

    model, _, effort = args.judge_model.partition("@")
    effort = effort or None
    base_url = os.environ.get("JUDGE_BASE_URL", "")
    api_key = os.environ.get("JUDGE_API_KEY", "")
    if not args.dry_run and not (base_url and api_key):
        print(
            "ERROR: JUDGE_BASE_URL and JUDGE_API_KEY must be set (or use --dry-run)",
            file=sys.stderr,
        )
        return 1

    rubrics_from = Path(args.rubrics_from) if args.rubrics_from else None
    if rubrics_from is not None and not rubrics_from.is_dir():
        print(f"ERROR: --rubrics-from dir not found: {rubrics_from}", file=sys.stderr)
        return 1

    gate_failures = 0
    empty_jobs = []
    for job in job_dirs:
        jd = Path(job)
        trials = [classify(t, rubrics_from) for t in sorted(jd.glob("*__*")) if t.is_dir()]
        n_trials = len(trials)
        # A dir with no `<task>__<hash>` children is almost always the wrong path — a TRIAL dir
        # rather than the JOB dir above it, or a typo. Without this it prints a tidy "0 trial dirs
        # ... scored 0/0" and exits 0, which reads as "this system has no process score" rather than
        # "you pointed me somewhere else". Same denominator rule as the coverage gate below.
        if n_trials == 0:
            empty_jobs.append(jd)
            continue
        if args.limit:
            trials = trials[: args.limit]

        todo, cached, skipped = [], [], []
        for rec in trials:
            if rec["reason"]:
                skipped.append(rec)
                continue
            out_p = rec["path"] / "verifier" / "process.json"
            rec["inputs_sha256"] = inputs_sha(rec)
            expect = {
                "scorer_version": SCORER_VERSION,
                "judge_model": model + (f"@{effort}" if effort else ""),
                # Same model NAME behind a different proxy can be a different deployment.
                "judge_base_url": base_url,
                "inputs_sha256": rec["inputs_sha256"],
            }
            if cached_ok(out_p, expect) and not args.force:
                cached.append(rec)
            else:
                todo.append(rec)

        print(
            f"\n=== {jd.name}: {n_trials} trial dirs — {len(todo)} to grade, {len(cached)} cached, {len(skipped)} skipped"
        )
        if args.limit:
            print(
                f"    ⚠️  PARTIAL RUN: --limit {args.limit} — coverage gate suppressed, numbers are NOT job-wide"
            )

        results: dict[str, dict] = {}
        if args.dry_run:
            print("    --dry-run: no grading performed")
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {
                    ex.submit(grade_one, rec, model, effort, base_url, api_key): rec for rec in todo
                }
                for i, f in enumerate(as_completed(futs), 1):
                    rec = futs[f]
                    payload = f.result()
                    (rec["path"] / "verifier").mkdir(parents=True, exist_ok=True)
                    (rec["path"] / "verifier" / "process.json").write_text(
                        json.dumps(payload, indent=2)
                    )
                    results[rec["trial"]] = payload
                    if args.amend_reward and not payload.get("error"):
                        amend_reward(rec["path"], payload)
                    if i % 25 == 0:
                        print(f"    {i}/{len(todo)} graded", file=sys.stderr)

        # Cached trials count toward coverage and the reported mean, so a re-run reports the
        # same job-wide number instead of only what it happened to grade this time.
        for rec in cached:
            try:
                results[rec["trial"]] = json.loads(
                    (rec["path"] / "verifier" / "process.json").read_text()
                )
            except (OSError, ValueError):
                results[rec["trial"]] = {"error": "unreadable cached process.json"}
        # Amend CACHED trials too. process.json is written on the first (un-amended) run, so a later
        # `--amend-reward` found every trial cached and silently amended nothing while exiting 0 —
        # and metric.py's mean_process_pct exists only for amended rewards, making the documented
        # recovery workflow a no-op.
        if args.amend_reward:
            n_amended = 0
            for rec in cached:
                payload = results.get(rec["trial"]) or {}
                if not payload.get("error") and amend_reward(rec["path"], payload):
                    n_amended += 1
            if n_amended:
                print(f"    amended {n_amended} cached trial reward.json")

        errored = [t for t, p in results.items() if p.get("error")]
        pcts = [p["score"]["pct"] for p in results.values() if p.get("score")]
        by_reason: dict[str, int] = {}
        for rec in skipped:
            by_reason[rec["reason"]] = by_reason.get(rec["reason"], 0) + 1

        n_scored = len(pcts)
        denom = n_trials if not args.limit else len(trials)
        print(
            f"    scored {n_scored}/{denom}"
            + (f"  mean process_pct = {sum(pcts) / n_scored:.1f}" if pcts else "")
        )
        if by_reason:
            print(f"    skipped: {by_reason}")
        if errored:
            print(
                f"    ⚠️  {len(errored)} judge error(s): {errored[:3]}{' …' if len(errored) > 3 else ''}"
            )

        summary = {
            "schema": "healthcare/process_summary.v1",
            "scorer_version": SCORER_VERSION,
            "job": jd.name,
            "judge_model": args.judge_model,
            "n_trial_dirs": n_trials,
            "n_scored": n_scored,
            "n_cached": len(cached),
            "n_judge_errors": len(errored),
            "skipped": by_reason,
            "partial_limit": args.limit or None,
            "mean_process_pct": (sum(pcts) / n_scored) if pcts else None,
        }
        if not args.dry_run:
            (jd / "process_summary.json").write_text(json.dumps(summary, indent=2))

        # The gate: only OUR failures count. no_report / no_process_rubrics are correct skips.
        # Derive from the shared FAILURE_REASONS rather than listing reasons here. A gate that
        # enumerates its own reasons drifts, and a reason left out means a job whose verifier
        # never ran exits 0 with an incomplete summary.
        bad = len(errored) + sum(by_reason.get(r, 0) for r in FAILURE_REASONS)
        if bad and not args.limit:
            gate_failures += bad

    if empty_jobs:
        print(
            "\n❌ no trial dirs (`<task>__<hash>/`) found in: "
            + ", ".join(str(d) for d in empty_jobs)
            + "\n   Pass the JOB dir, not a trial dir — e.g. jobs/<job>, not jobs/<job>/<task>__<hash>.",
            file=sys.stderr,
        )
        return 2
    if gate_failures:
        print(
            f"\n❌ coverage gate: {gate_failures} trial(s) could not be scored for a fixable reason "
            f"(unreadable rubrics or judge error). Re-run to retry, or pass --allow-incomplete.",
            file=sys.stderr,
        )
        return 0 if args.allow_incomplete else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
