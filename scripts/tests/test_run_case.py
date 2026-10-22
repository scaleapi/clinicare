"""Tests for run_case.py — the one-task execution contract that wrappers (MedHELM) depend on.

What a wrapper relies on, and therefore what must not drift silently:

  * `status` separates a scored trial ("ok"), a scored trial whose agent raised ("agent_error"),
    and an UNSCORED trial ("dead"). Harbor exits 0 in all three, so this field is the only signal;
    reporting a dead trial as scored would turn an infrastructure failure into a 0.0 answer.
  * `task_id` comes from the trial's task path, never its dir name: harbor truncates trial dir
    names to 32 characters, so the name is not a reliable task id.
  * run_case.py builds exactly run_sweep.py's command for the system, plus only `-i` and `-o`.
  * usage errors exit 2 before anything runs.
  * .env is read as the shell reads it: run_case.py loads it itself (load_env), and a trailing
    `# comment` kept in a value once crashed every codex trial (int('120   # codex (seconds)')).
  * a trial is reported only if harbor ran the job and the trial's task matches the current task
    dir: harbor's resume compares settings, not task content, so an old trial would otherwise be
    served as fresh after a CliniCARE change or a task rebuild.

Offline: fixtures are synthetic (placeholder ids), no harbor, no Docker, no credentials.

    python scripts/tests/test_run_case.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from dirhash import dirhash

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import run_case
import run_sweep
from clinicare_core import load_env

_fails: list[str] = []
# Longer than harbor's 32-char trial-name cap, to prove task_id is not read from the dir name.
LONG_ID = "placeholder-scenario-with-a-long-name-000001"


def check(label: str, cond: bool, detail: str = "") -> None:
    print(
        f"  {'ok  ' if cond else 'FAIL'}  {label}"
        + (f"   [{detail}]" if detail and not cond else "")
    )
    if not cond:
        _fails.append(label)


def make_trial(
    job: Path,
    task_id: str,
    *,
    rewards: bool,
    reward_file: bool,
    exc: str | None,
    checksum: str | None = None,
    reward: dict | None = None,
):
    trial = job / f"{task_id[:32].rstrip('_-')}__abcdefg"
    (trial / "verifier").mkdir(parents=True)
    result = {
        "task_name": f"mimic/{task_id}",
        "task_id": {"path": f"/somewhere/benchmark/tasks/{task_id}"},
        "config": {"agent": {"name": "pkg.mod:Agent", "model_name": "openai/placeholder"}},
        "agent_info": {"version": "1.2.3"},
        "verifier_result": {"rewards": {"reward": 1.0}} if rewards else None,
        "exception_info": {"exception_type": exc} if exc else None,
        "task_checksum": checksum,
    }
    (trial / "result.json").write_text(json.dumps(result))
    if reward_file:
        (trial / "verifier" / "reward.json").write_text(json.dumps(reward or {"score": 1.0}))
    return trial


def run_main_with_fake_harbor(
    *,
    harbor_rc: int,
    stale: bool,
    corrupt: str | None = None,
    grade: bool = False,
    reward: dict | None = None,
    judges: dict | None = None,
    judge_env: bool = True,
    old_summaries: dict | None = None,
    backdate: bool = False,
) -> tuple[int, str]:
    """run_case.main() against a placeholder task, with preflight and harbor replaced by fakes.

    `judges` fakes each judge script as {"process"|"policy": (exit code, summary, trial record)}:
    the summary and record are what that script writes to the job dir and the trial dir.
    `backdate` stamps each new summary with a time before this run, as a skewed NFS clock would.
    """
    task_id = "advanced-imaging-000001"
    with tempfile.TemporaryDirectory() as tmp:
        tasks = Path(tmp, "tasks")
        (tasks / task_id).mkdir(parents=True)
        (tasks / task_id / "task.toml").write_text("[task]\n")
        jobs = Path(tmp, "jobs")
        checksum = "stale" if stale else dirhash(tasks / task_id, "sha256")
        # The trial harbor would leave (or that an earlier run left) in the job dir.
        trial = make_trial(
            jobs / "jn",
            task_id,
            rewards=True,
            reward_file=True,
            exc=None,
            checksum=checksum,
            reward=reward,
        )
        if corrupt:
            (trial / corrupt).write_text('{"score": 1.')
        # Files an earlier grading run left in the job and trial dirs: {judge: (summary, record)}.
        for name, (summary, record) in (old_summaries or {}).items():
            _, trial_file, summary_file = run_case.JUDGES[name]
            (jobs / "jn" / summary_file).write_text(json.dumps(summary))
            (trial / "verifier" / trial_file).write_text(json.dumps(record))
            old = time.time() - 3600
            os.utime(jobs / "jn" / summary_file, (old, old))

        def fake_run(cmd, **kwargs):
            script = next((Path(p).name for p in cmd if "benchmark/eval/metrics/" in str(p)), None)
            if script is None:
                return subprocess.CompletedProcess(cmd, harbor_rc)
            name = next(n for n, (s, _, _) in run_case.JUDGES.items() if s == script)
            _, trial_file, summary_file = run_case.JUDGES[name]
            code, summary, record = (judges or {}).get(name, (0, None, None))
            if summary is not None:
                (jobs / "jn" / summary_file).write_text(json.dumps(summary))
                if backdate:
                    past = time.time() - 1800
                    os.utime(jobs / "jn" / summary_file, (past, past))
            if record is not None:
                (trial / "verifier" / trial_file).write_text(json.dumps(record))
            return subprocess.CompletedProcess(cmd, code)

        child = {"JUDGE_BASE_URL": "https://judge.example", "JUDGE_API_KEY": "placeholder"}

        argv = ["run_case.py", "--system", "codex-gpt56sol", "--task-id", task_id]
        argv += ["--tasks", str(tasks), "--jobs-dir", str(jobs), "--job-name", "jn"]
        argv += ["--grade"] if grade else []
        out, err = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(run_sweep, "preflight", return_value=0),
            mock.patch.object(run_sweep, "child_env", return_value=child if judge_env else {}),
            # build_cmd forwards the system's key; supply a placeholder so the test does not
            # depend on a developer's .env (a clean checkout or CI has none).
            mock.patch.dict(run_sweep.os.environ, {"OPENAI_API_KEY": "placeholder"}),
            mock.patch.object(run_case.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            rc = run_case.main()
        return rc, out.getvalue()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        print("== 1. status classification ==")
        for name, kw, want in [
            ("ok", {"rewards": True, "reward_file": True, "exc": None}, "ok"),
            (
                "agent raised, verifier ran",
                {"rewards": True, "reward_file": True, "exc": "AgentTimeoutError"},
                "agent_error",
            ),
            (
                "no verifier result",
                {"rewards": False, "reward_file": False, "exc": "RuntimeError"},
                "dead",
            ),
            (
                "rewards but no reward.json",
                {"rewards": True, "reward_file": False, "exc": None},
                "dead",
            ),
            # The verifier's own failure paths still write score 0.0: unscored, not a wrong verdict.
            (
                "rubric bundle failed to load (judge.py)",
                {
                    "rewards": True,
                    "reward_file": True,
                    "exc": None,
                    "reward": {"score": 0.0, "judge_failed": 1, "rubrics_load_failed": 1},
                },
                "dead",
            ),
            (
                "judge failed to launch (test.sh)",
                {
                    "rewards": True,
                    "reward_file": True,
                    "exc": None,
                    "reward": {"score": 0.0, "judge_failed": 1, "judge_launch_failed": 1},
                },
                "dead",
            ),
            # ...whereas a missing report is a real 0, and judge_failed alone (process pass) is fine.
            (
                "agent wrote no report",
                {
                    "rewards": True,
                    "reward_file": True,
                    "exc": None,
                    "reward": {"score": 0.0, "report_present": 0, "report_chars": 0},
                },
                "ok",
            ),
            (
                "only the process judge failed",
                {
                    "rewards": True,
                    "reward_file": True,
                    "exc": None,
                    "reward": {"score": 1.0, "outcome_score": 1.0, "judge_failed": 1},
                },
                "ok",
            ),
        ]:
            job = root / name.replace(" ", "_").replace("/", "_")
            make_trial(job, "advanced-imaging-000001", **kw)
            got = run_case.trial_report(job)
            check(f"{name} -> {want}", got is not None and got["status"] == want, str(got))

        print("\n== 2. task id comes from the task path, not the truncated dir name ==")
        job = root / "long"
        trial = make_trial(job, LONG_ID, rewards=True, reward_file=True, exc=None)
        got = run_case.trial_report(job) or {}
        check("dir name really is truncated", not trial.name.startswith(LONG_ID))
        check("task_id is the full id", got.get("task_id") == LONG_ID, str(got.get("task_id")))
        check(
            "model/agent/version carried",
            (got.get("model_name"), got.get("agent_name"), got.get("agent_version"))
            == ("openai/placeholder", "pkg.mod:Agent", "1.2.3"),
        )

        print("\n== 3. no single trial -> None (caller reports 'harbor produced no trial') ==")
        check("empty job dir", run_case.trial_report(root / "missing") is None)
        job = root / "two"
        make_trial(job, "advanced-imaging-000001", rewards=True, reward_file=True, exc=None)
        make_trial(job, "advanced-imaging-000002", rewards=True, reward_file=True, exc=None)
        check("two trials", run_case.trial_report(job) is None)

    print("\n== 4. the command is run_sweep's, plus only -i/-o ==")
    system = next(s for s in run_sweep.SYSTEMS if s.tag == "codex-gpt56sol")
    base = run_sweep.build_cmd(system, "benchmark/tasks", 1, 1, "docker", dry_run=True)
    one = run_sweep.build_cmd(
        system,
        "benchmark/tasks",
        1,
        1,
        "docker",
        dry_run=True,
        job="jn",
        extra=("-i", "advanced-imaging-000001", "-o", "/tmp/jobs"),
    )
    jn_at = base.index("--job-name") + 1
    check("default job name unchanged", base[jn_at] == "cbench_sweep_codex-gpt56sol")
    check("job overrides the name", one[jn_at] == "jn")
    check(
        "only -i/-o added",
        one[:jn_at] + one[jn_at + 1 : len(base)] == base[:jn_at] + base[jn_at + 1 :]
        and one[len(base) :] == ["-i", "advanced-imaging-000001", "-o", "/tmp/jobs"],
    )

    print("\n== 5. usage errors exit 2 before running anything ==")
    for label, argv in [
        ("unknown system", ["--system", "no-such-system", "--task-id", "x"]),
        ("unknown task", ["--system", "codex-gpt56sol", "--task-id", "advanced-imaging-000001"]),
    ]:
        rc = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "run_case.py"),
                *argv,
                "--jobs-dir",
                "/tmp/j",
                "--job-name",
                "n",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        check(f"{label} -> exit 2", rc == 2, f"exit {rc}")

    print("\n== 6. .env values match `set -a; source .env` ==")
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, ".env").write_text(
            "A_TRAILING=120   # comment\n"
            "B_PLAIN=value\n"
            'C_DOUBLE="has # inside"\n'
            "D_SINGLE='x y'\n"
            "E_HASH_NO_SPACE=abc#def\n"
            "F_REF=${B_PLAIN}/v1\n"
            'G_QUOTED_THEN_COMMENT="https://gw.example.com/v1"   # gateway\n'
            "H_EMPTY_THEN_COMMENT=   # leave blank\n"
            "I_SINGLE_THEN_COMMENT='abc'  # note\n"
            'J_ESCAPED_QUOTE="pa\\"ss"\n'
            'K_ESCAPED_BACKSLASH="a\\\\b"\n'
            'L_OTHER_BACKSLASH="a\\nb"\n'
            'M_ESCAPED_THEN_COMMENT="x\\"y"   # note\n'
            "N_SINGLE_BACKSLASH='a\\b'\n"
        )
        env = load_env(Path(tmp))
        for key, want in [
            ("A_TRAILING", "120"),
            ("B_PLAIN", "value"),
            ("C_DOUBLE", "has # inside"),
            ("D_SINGLE", "x y"),
            ("E_HASH_NO_SPACE", "abc#def"),
            ("F_REF", "value/v1"),
            ("G_QUOTED_THEN_COMMENT", "https://gw.example.com/v1"),
            ("H_EMPTY_THEN_COMMENT", ""),
            ("I_SINGLE_THEN_COMMENT", "abc"),
            # Expected values are what bash's `set -a; source .env` produced for these lines.
            ("J_ESCAPED_QUOTE", 'pa"ss'),
            ("K_ESCAPED_BACKSLASH", "a\\b"),
            ("L_OTHER_BACKSLASH", "a\\nb"),
            ("M_ESCAPED_THEN_COMMENT", 'x"y'),
            ("N_SINGLE_BACKSLASH", "a\\b"),
        ]:
            check(f"{key} == {want!r}", env.get(key) == want, repr(env.get(key)))

    print("\n== 7. only a trustworthy trial is reported (main(), harbor faked) ==")
    for label, harbor_rc, stale, corrupt, want_rc in [
        ("fresh trial", 0, False, None, 0),
        ("harbor refused the job; an old trial is in the dir", 1, False, None, 2),
        ("resumed trial built from an older version of the task", 0, True, None, 2),
        ("malformed reward.json", 0, False, "verifier/reward.json", 2),
        ("malformed result.json", 0, False, "result.json", 2),
    ]:
        rc, out = run_main_with_fake_harbor(harbor_rc=harbor_rc, stale=stale, corrupt=corrupt)
        check(f"{label} -> exit {want_rc}", rc == want_rc, f"exit {rc}")
        check(f"{label}: JSON only on success", bool(out.strip()) == (want_rc == 0), repr(out))
        if want_rc == 0:
            report = json.loads(out)
            check(
                "fresh trial: status ok, not graded",
                (report.get("status"), report.get("graded")) == ("ok", False),
            )

    print("\n== 8. --grade: each judge's outcome and reason, and graded ==")
    scored = {"n_scored": 1, "n_judge_errors": 0, "skipped": {}}

    def skip(reason: str) -> dict:
        return {"n_scored": 0, "n_judge_errors": 0, "skipped": {reason: 1}}

    def error(text: str) -> tuple[int, dict, dict]:
        return (2, {"n_scored": 0, "n_judge_errors": 1, "skipped": {}}, {"error": text})

    cases = [
        # label, judges, judge env set, want (process reason, policy reason), want graded
        (
            "both judges scored",
            {"process": (0, scored, {}), "policy": (0, scored, {})},
            True,
            ("scored", "scored"),
            True,
        ),
        (
            "policy-off task",
            {"process": (0, scored, {}), "policy": (0, skip("policy_off"), None)},
            True,
            ("scored", "policy_off"),
            True,
        ),
        (
            "agent wrote no report",
            {"process": (0, skip("no_report"), None), "policy": (0, skip("no_report"), None)},
            True,
            ("no_report", "no_report"),
            True,
        ),
        (
            "judge keys not set",
            {},
            False,
            ("judge_credentials_missing", "judge_credentials_missing"),
            False,
        ),
        (
            "judge API rejected the key",
            {
                "process": error("AuthenticationError: Error code: 401 - invalid api key"),
                "policy": error("PermissionDeniedError: Error code: 403 - key not allowed"),
            },
            True,
            ("judge_credentials_rejected", "judge_credentials_rejected"),
            False,
        ),
        # Mentions "401" and "api key" but is a rate limit: matched by exception type, not text.
        (
            "judge rate-limited",
            {
                "process": error(
                    "RateLimitError: Error code: 429 - retry after 401ms; api key quota"
                ),
                "policy": (0, scored, {}),
            },
            True,
            ("judge_error", "scored"),
            False,
        ),
        (
            "rubric bundle missing",
            {"process": (2, skip("no_rubrics"), None), "policy": (0, scored, {})},
            True,
            ("no_rubrics", "scored"),
            False,
        ),
        (
            "judge script crashed",
            {"process": (1, None, None), "policy": (0, scored, {})},
            True,
            ("judge_script_failed", "scored"),
            False,
        ),
    ]
    for label, judges, judge_env, want_reasons, want_graded in cases:
        rc, out = run_main_with_fake_harbor(
            harbor_rc=0, stale=False, grade=True, judges=judges, judge_env=judge_env
        )
        report = json.loads(out) if rc == 0 else {}
        grading = report.get("grading") or {}
        got = tuple((grading.get(j) or {}).get("reason") for j in ("process", "policy"))
        check(f"{label}: reasons {want_reasons}", got == want_reasons, f"exit {rc}, {got}")
        check(
            f"{label}: graded {want_graded}",
            report.get("graded") is want_graded,
            str(report.get("graded")),
        )
    # An earlier run's summary says "scored", but this run's script failed: never report it scored.
    rc, out = run_main_with_fake_harbor(
        harbor_rc=0,
        stale=False,
        grade=True,
        judges={"process": (2, scored, None), "policy": (0, scored, {})},
    )
    got = (json.loads(out)["grading"]["process"]["reason"], json.loads(out)["graded"])
    check(
        "stale 'scored' summary, script failed -> judge_script_failed",
        got == ("judge_script_failed", False),
        str(got),
    )
    # An earlier run's key was rejected; this run the policy judge stops before writing a summary
    # (e.g. a missing corpus). Report this run's failure, not last run's cause.
    rejected = {"n_scored": 0, "n_judge_errors": 1, "skipped": {}}
    rc, out = run_main_with_fake_harbor(
        harbor_rc=0,
        stale=False,
        grade=True,
        judges={"process": (0, scored, {}), "policy": (1, None, None)},
        old_summaries={"policy": (rejected, {"error": "AuthenticationError: Error code: 401"})},
    )
    got = json.loads(out)["grading"]["policy"]["reason"]
    check("old run's summary, no new one -> judge_script_failed", got == "judge_script_failed", got)
    # The summary is rewritten, but the file server's clock is behind ours: it still counts.
    rc, out = run_main_with_fake_harbor(
        harbor_rc=0,
        stale=False,
        grade=True,
        judges={"process": (0, scored, {}), "policy": (0, scored, {})},
        old_summaries={"process": (rejected, {}), "policy": (rejected, {})},
        backdate=True,
    )
    got = (json.loads(out)["grading"]["process"]["outcome"], json.loads(out)["graded"])
    check("rewritten summary with a lagging mtime -> scored", got == ("scored", True), str(got))
    rc, out = run_main_with_fake_harbor(harbor_rc=0, stale=False)
    check("without --grade: grading is null", json.loads(out).get("grading", "absent") is None, out)

    print()
    if _fails:
        print(f"{len(_fails)} FAILURE(S): " + "; ".join(_fails))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
