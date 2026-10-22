"""Tests for the unscored-reward rule in score_clinicare.py (the leaderboard / paper tables).

When the in-container verifier fails before scoring (rubric bundle won't load, or judge.py won't
launch), it still writes reward.json with `score: 0.0` so harbor has a reward to read. That 0.0 is
not a verdict. score_clinicare must count such a trial with the infrastructure failures
(`missing_reward`), exactly like a trial with no reward.json, and never as a wrong answer, or it
would lower a system's accuracy for a case nobody checked. The rule lives in
clinicare_core.rewards; these checks pin that score_clinicare applies it.

    python benchmark/eval/tests/test_unscored_rewards.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from clinicare_core import VERIFIER_FAILED_KEYS, verifier_failed, write_reward_atomic
from clinicare_core.paths import bootstrap

bootstrap(__file__, "report")
import score_clinicare as sc

FAIL: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def _trial(job: str, task: str, reward: dict) -> str:
    d = os.path.join(job, f"{task}__x")
    os.makedirs(os.path.join(d, "verifier"), exist_ok=True)
    with open(os.path.join(d, "verifier/reward.json"), "w") as f:
        json.dump(reward, f)
    return d


def main() -> int:
    print("== the rule ==")
    check("rubrics_load_failed", verifier_failed({"score": 0.0, "rubrics_load_failed": 1}), True)
    check("judge_launch_failed", verifier_failed({"score": 0.0, "judge_launch_failed": 1}), True)
    check(
        "judge_failed alone (process pass)",
        verifier_failed({"score": 1.0, "judge_failed": 1}),
        False,
    )
    check("no report (a genuine 0)", verifier_failed({"score": 0.0, "report_present": 0}), False)

    with tempfile.TemporaryDirectory() as tmp:
        job = os.path.join(tmp, "job")
        scored = {"outcome_score": 1.0, "outcome_label_parsed": 1, "report_present": 1}
        print("\n== score_clinicare.load_trial ==")
        check(
            "scored trial loads", sc.load_trial(_trial(job, "a-000001", scored)) is not None, True
        )
        no_report = {"score": 0.0, "report_present": 0, "report_chars": 0}
        nr = sc.load_trial(_trial(job, "a-000002", no_report))
        check("no-report trial loads, scored 0", (nr or {}).get("score"), 0)
        for i, key in enumerate(VERIFIER_FAILED_KEYS):
            d = _trial(job, f"a-00001{i}", {"score": 0.0, "judge_failed": 1, key: 1})
            check(f"{key} trial is dropped (unscored)", sc.load_trial(d), None)

        print("\n== score_job: unscored trials count as missing_reward, not as wrong ==")
        row = sc.score_job(job, set())
        check("n_dirs", row["n_dirs"], 2 + len(VERIFIER_FAILED_KEYS))
        check("scored trials", row["n"], 2)
        check("missing_reward", row["missing_reward"], len(VERIFIER_FAILED_KEYS))
        check("accuracy over scored trials only", row["acc"], 50.0)

    print("\n== write_reward_atomic: a concurrent reader never sees a partial reward.json ==")
    with tempfile.TemporaryDirectory() as tmp:
        rp = os.path.join(tmp, "reward.json")
        write_reward_atomic(rp, {"score": 1.0})
        with open(rp) as f:
            check("writes the reward", json.load(f), {"score": 1.0})
        check("leaves no temp file", sorted(os.listdir(tmp)), ["reward.json"])
        # A shared job: the rewrite keeps the file readable to the other users who read it.
        os.chmod(rp, 0o644)
        write_reward_atomic(rp, {"score": 0.0})
        check("keeps the original file mode", oct(os.stat(rp).st_mode & 0o777), oct(0o644))
        # A grader amending the reward while a run_case.py sharing the job reads it.
        big = {"score": 1.0, **{f"key_{i}": i for i in range(2000)}}
        bad: list[str] = []
        done = threading.Event()

        def reader() -> None:
            while not done.is_set():
                try:
                    with open(rp) as f:
                        json.load(f)
                except ValueError as exc:
                    bad.append(str(exc))

        thread = threading.Thread(target=reader)
        thread.start()
        for i in range(300):
            write_reward_atomic(rp, {**big, "process_pct": i / 300})
        done.set()
        thread.join()
        check("reader saw no partial file across 300 rewrites", len(bad), 0)
        check("still no temp files", sorted(os.listdir(tmp)), ["reward.json"])

    print("\n== benchmark/metric.py: unscored rewards stay out of the job means ==")
    metric_py = Path(__file__).resolve().parents[2] / "metric.py"
    spec = importlib.util.spec_from_file_location("job_metric", metric_py)
    job_metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job_metric)
    check(
        "VERIFIER_FAILED_KEYS matches clinicare_core",
        job_metric.VERIFIER_FAILED_KEYS,
        VERIFIER_FAILED_KEYS,
    )
    with tempfile.TemporaryDirectory() as tmp:
        lines = [{"score": 1.0}, {"score": 0.0}, None]
        lines += [{"score": 0.0, "judge_failed": 1, key: 1} for key in VERIFIER_FAILED_KEYS]
        src, dst = Path(tmp, "rewards.jsonl"), Path(tmp, "metric.json")
        src.write_text("\n".join("null" if r is None else json.dumps(r) for r in lines) + "\n")
        subprocess.run([sys.executable, str(metric_py), "-i", str(src), "-o", str(dst)], check=True)
        out = json.loads(dst.read_text())
        check("n_trials counts every line", out["n_trials"], 3 + len(VERIFIER_FAILED_KEYS))
        check("n_completed counts scored trials only", out["n_completed"], 2)
        check("n_verifier_failed", out["n_verifier_failed"], len(VERIFIER_FAILED_KEYS))
        check("mean_score over scored trials only", out["mean_score"], 0.5)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): " + "; ".join(FAIL))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
