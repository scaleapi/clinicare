"""Run ONE CliniCARE task with one system, into a caller-chosen job dir, and report the trial as JSON.

This is the stable execution command for wrappers (e.g. MedHELM): they pass a system tag and a task
id, and read one JSON line back. Everything about HOW a system runs (agent shim, model, web-off
flags, credential forwarding, preflights) comes from run_sweep.py, so a wrapper never duplicates
it and picks up CliniCARE changes for free.

    uv run python scripts/run_case.py --system codex-gpt56sol --task-id <task id> \\
        --jobs-dir jobs/medhelm --job-name <unique name>

.env is loaded here (real environment wins), so callers need not `set -a; source .env`.

Output contract — exit 0 and one JSON object on stdout whenever a trial dir exists:
    status          "ok"          the verifier ran and the agent raised nothing
                    "agent_error" the agent raised (e.g. timeout, non-zero exit) but the verifier ran
                    "dead"        unscored: no verifier result (setup/environment failure), or the
                                  verifier itself failed before scoring (clinicare_core.rewards)
    trial_dir       absolute path of <jobs-dir>/<job-name>/<task>__<suffix>/
    task_id         the task id (from the trial's task path, not its truncated dir name)
    model_name      config.agent.model_name of the trial      agent_name  config.agent.name
    agent_version   agent_info.version                         exception_type  or null
    graded          whether --grade ran and both judges finished: each one scored the trial or skipped
                    it by design (no report, no rubrics, policy-off task). False without --grade, and
                    false when a judge could not score it (judge error, missing judge credentials)
    grading         null without --grade; else {"process": J, "policy": J}, one per judge, where J is
                    {"outcome": "scored" | "skipped" | "failed", "reason": str, "detail": str | null}
                    so a wrapper can say what went wrong and where to look. Reasons by outcome:
                      skipped  no_report (the agent wrote none), no_process_rubrics,
                               no_policy_rubrics, policy_off (nothing to score, by design)
                      failed   judge_credentials_missing (JUDGE_BASE_URL / JUDGE_API_KEY unset),
                               judge_credentials_rejected (the judge API returned 401 or 403),
                               judge_error (any other judge failure; detail has the exception),
                               no_rubrics (the task's rubric bundle is missing or unreadable),
                               judge_script_failed (the script exited non-zero with no record)
Exit 2 with a message on stderr when there is no trustworthy trial to report: a usage/preflight
error (nothing ran), harbor refusing the job (e.g. the job dir exists with different settings), a
trial whose result.json or reward.json cannot be parsed, or a finished trial in the job dir that
was built from a different version of the task.

Re-running with the same --job-name resumes: harbor keeps a finished trial and returns it without
re-running. That trial is reported only if its task is byte-identical to the current task dir
(harbor's task checksum); after a CliniCARE change or a task rebuild, use a new --job-name.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from dirhash import dirhash  # harbor's own task checksum (Task.checksum == dirhash(dir, sha256))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sweep
from clinicare_core import load_env, verifier_failed

BASE = run_sweep.BASE
EXIT_USAGE = 2

# The two post-hoc judges: name in the `grading` report -> (script, per-trial file, job summary).
JUDGES = {
    "process": ("process.py", "process.json", "process_summary.json"),
    "policy": ("policy_support.py", "policy_support.json", "policy_support_summary.json"),
}
JUDGE_CREDENTIALS = ("JUDGE_BASE_URL", "JUDGE_API_KEY")
# What the openai SDK raises when the judge API refuses the key (401 / 403). The judges record an
# error as "<ExceptionType>: <message>", so the type name is matched, never the message text.
_REJECTED_KEY_ERRORS = ("AuthenticationError", "PermissionDeniedError")
_FAILED_SKIPS = ("no_rubrics", "unverified")  # process.py FAILURE_REASONS that are skip records


def fail(msg: str) -> int:
    print(f"run_case: {msg}", file=sys.stderr)
    return EXIT_USAGE


class UnreadableTrial(Exception):
    """The trial's result.json or reward.json is not valid JSON, so nothing about it can be trusted."""


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise UnreadableTrial(f"cannot read {path}: {exc}") from exc


def trial_report(job_dir: Path) -> dict | None:
    """The contract JSON for the single trial in `job_dir`, or None if harbor left no trial.

    Raises UnreadableTrial if the trial's result.json or reward.json cannot be parsed.
    """
    trials = [d for d in sorted(job_dir.glob("*__*")) if (d / "result.json").is_file()]
    if len(trials) != 1:
        return None
    trial = trials[0]
    result = _read_json(trial / "result.json")
    agent = result.get("config", {}).get("agent", {})
    exception = result.get("exception_info") or {}
    verified = (
        bool((result.get("verifier_result") or {}).get("rewards"))
        and (trial / "verifier" / "reward.json").is_file()
    )
    if verified:
        reward = _read_json(trial / "verifier" / "reward.json")
        verified = not verifier_failed(reward)  # its 0.0 is not a verdict
    status = "dead" if not verified else ("agent_error" if exception else "ok")
    task_path = (result.get("task_id") or {}).get("path") or ""
    return {
        "status": status,
        "trial_dir": str(trial.resolve()),
        "task_id": Path(task_path).name or result.get("task_name", "").split("/")[-1],
        "model_name": agent.get("model_name"),
        "agent_name": agent.get("name"),
        "agent_version": (result.get("agent_info") or {}).get("version"),
        "exception_type": exception.get("exception_type"),
        "task_checksum": result.get("task_checksum"),
    }


def log_tail(log: Path, lines: int = 5) -> str:
    try:
        return " | ".join(log.read_text().strip().splitlines()[-lines:])
    except OSError:
        return "(no log)"


def _judge_outcome(outcome: str, reason: str, detail: str | None = None) -> dict:
    return {"outcome": outcome, "reason": reason, "detail": detail}


def _fingerprint(path: Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def judge_result(
    job_dir: Path, trial_dir: Path, judge: str, exit_code: int, before: tuple | None
) -> dict:
    """What one judge did with this job's single trial, read from the files the judge wrote.

    Only a summary this run wrote counts: `before` is its fingerprint from before the judges ran.
    An earlier run's summary describes that run: a judge that now stops before writing one (a
    missing corpus, a crash) would otherwise be reported with the old cause, e.g. a key that was
    rejected last time. Comparing against this process's clock instead would misfire on EFS, where
    the NFS server stamps mtime from its own clock.
    """
    _, trial_file, summary_file = JUDGES[judge]
    summary_path = job_dir / summary_file
    try:
        if before is not None and _fingerprint(summary_path) == before:
            return _judge_outcome(
                "failed", "judge_script_failed", f"exit {exit_code}, wrote no new {summary_file}"
            )
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return _judge_outcome(
            "failed", "judge_script_failed", f"exit {exit_code}, no {summary_file}"
        )
    skipped = summary.get("skipped") or {}
    if skipped:
        reason = next(iter(skipped))
        return _judge_outcome("failed" if reason in _FAILED_SKIPS else "skipped", reason)
    if summary.get("n_judge_errors"):
        try:
            error = json.loads((trial_dir / "verifier" / trial_file).read_text()).get("error") or ""
        except (OSError, ValueError):
            error = ""
        kind = error.split(":", 1)[0].strip()
        reason = "judge_credentials_rejected" if kind in _REJECTED_KEY_ERRORS else "judge_error"
        return _judge_outcome("failed", reason, error or None)
    if summary.get("n_scored") or summary.get("n_cached"):
        return _judge_outcome("scored", "scored")
    return _judge_outcome("failed", "judge_script_failed", f"exit {exit_code}, nothing scored")


def grade_case(job_dir: Path, trial_dir: Path, env: dict[str, str]) -> dict:
    """Run both judges on the case (unless their credentials are missing) and report each one."""
    missing = [key for key in JUDGE_CREDENTIALS if not env.get(key)]
    if missing:
        detail = f"{', '.join(missing)} not set in the environment or .env"
        return {
            judge: _judge_outcome("failed", "judge_credentials_missing", detail) for judge in JUDGES
        }
    results = {}
    before = {judge: _fingerprint(job_dir / JUDGES[judge][2]) for judge in JUDGES}
    for judge, code in zip(JUDGES, grade(job_dir, env)):
        result = judge_result(job_dir, trial_dir, judge, code, before[judge])
        if code != 0 and result["outcome"] != "failed":
            # The summary is from an earlier run (a cached trial); this run's script failed.
            result = _judge_outcome(
                "failed", "judge_script_failed", f"exit {code}; see the job log"
            )
        results[judge] = result
    return results


def grade(job_dir: Path, env: dict[str, str]) -> list[int]:
    """Add the LLM-judged scores (process_pct, policy_citation_support_rate) to reward.json.

    grade.py writes side files only; these two scripts are what amend reward.json. Each has a
    coverage gate: exit 0 means every trial was scored or skipped by design (BENIGN_REASONS in
    process.py: no report, no rubrics, policy-off), exit 2 means one could not be scored for a
    fixable reason (FAILURE_REASONS: judge error, unreadable rubrics). So a clean exit from both is
    the right test. Checking reward.json for both judges' keys instead would mark every trial a
    judge correctly skips as ungraded. Returns each script's exit code, in JUDGES order.
    """
    return [
        subprocess.run(
            [run_sweep.PY, f"benchmark/eval/metrics/{script}", str(job_dir), "--amend-reward"],
            env=env,
            stdout=sys.stderr,
            check=False,
        ).returncode
        for script, _, _ in JUDGES.values()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--system", required=True, help="one System tag from run_sweep.py")
    ap.add_argument("--task-id", required=True, help="task dir name under --tasks")
    ap.add_argument("--jobs-dir", required=True, help="where harbor writes the job dir")
    ap.add_argument("--job-name", required=True, help="unique per (system, settings, task)")
    ap.add_argument("--tasks", default="benchmark/tasks", help="built task dir")
    ap.add_argument("--backend", choices=["docker", "modal"], default="docker")
    ap.add_argument("--effort", choices=["minimal", "low", "medium", "high", "xhigh"])
    ap.add_argument("--max-retries", type=int, default=None)
    ap.add_argument("--grade", action="store_true", help="run the LLM judges (needs JUDGE_*)")
    ap.add_argument("--dry-run", action="store_true", help="print the harbor command only")
    args = ap.parse_args()

    # run_sweep's helpers read os.environ, so merge .env into it (the real environment wins).
    run_sweep.os.environ.update(load_env(BASE))
    run_sweep.os.chdir(BASE)

    systems = [s for s in run_sweep.SYSTEMS if s.tag == args.system]
    if not systems:
        return fail(f"unknown system {args.system!r}; see SYSTEMS in scripts/run_sweep.py")
    system = systems[0]
    if not (BASE / args.tasks / args.task_id / "task.toml").is_file():
        return fail(f"task {args.task_id!r} not found under {args.tasks}")
    if not args.dry_run and run_sweep.preflight(systems, args.tasks, args.backend):
        return EXIT_USAGE
    env = run_sweep.child_env(args.backend)
    if env is None:
        return EXIT_USAGE

    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir = jobs_dir if jobs_dir.is_absolute() else BASE / jobs_dir
    cmd = run_sweep.build_cmd(
        system,
        args.tasks,
        k=1,
        n=1,
        backend=args.backend,
        effort=args.effort,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        job=args.job_name,
        extra=("-i", args.task_id, "-o", str(jobs_dir)),
    )
    if args.dry_run:
        shown = [c.split("=")[0] + "=***" if "_API_KEY=" in c else c for c in cmd]
        print(" ".join(shlex.quote(c) for c in shown))
        return 0

    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_dir = jobs_dir / args.job_name
    log_path = jobs_dir / f"{args.job_name}.log"
    with open(log_path, "a") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    # harbor exits 0 whatever the trial's outcome, so a non-zero code means harbor itself refused,
    # e.g. an existing job dir with different settings. Any trial already in that dir is from the
    # old settings: report the refusal, never that trial.
    if rc.returncode != 0:
        return fail(f"harbor refused the job (exit {rc.returncode}): {log_tail(log_path)}")
    try:
        report = trial_report(job_dir)
    except UnreadableTrial as exc:
        return fail(str(exc))
    if report is None:
        return fail(f"harbor produced no trial; see {log_path}")
    # A resumed trial whose task has since changed (CliniCARE update, task rebuild in place) would
    # otherwise be served as a fresh result: harbor's resume compares settings, not task content.
    current = dirhash(BASE / args.tasks / args.task_id, "sha256")
    if report["task_checksum"] != current:
        return fail(
            f"the trial in {job_dir} was built from a different version of task {args.task_id} "
            "(task checksum differs); use a new --job-name, or delete that job dir to re-run"
        )
    grading = None
    if args.grade and report["status"] != "dead":
        grading = grade_case(job_dir, Path(report["trial_dir"]), env)
    report["graded"] = grading is not None and all(
        j["outcome"] != "failed" for j in grading.values()
    )
    report["grading"] = grading
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
