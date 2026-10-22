"""Tests for passk_from_bundles.py -- the metric math and the refusals.

Fixtures are synthetic bundles built in a temp dir, with values chosen so every expected number is
hand-computable from the definitions in the script's docstring. No real bundles are needed, so this
runs anywhere in about a second.

Checking against hand-computed values rather than against another implementation is deliberate: two
implementations agreeing proves they agree, not that either is right. `--validate` covers the
agreement question separately.

    python3 benchmark/eval/tests/test_passk_from_bundles.py
    pytest benchmark/eval/tests/test_passk_from_bundles.py   # also collects
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from clinicare_core import VERIFIER_FAILED_KEYS
from clinicare_core.paths import bootstrap

bootstrap(__file__, "report")
import passk_from_bundles as pfb  # noqa: E402

FAIL: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}\n          got  {got!r}\n          want {want!r}")


def check_raises(name: str, fn) -> None:
    """A guard that cannot fire is worse than no guard, so assert the refusals too."""
    try:
        fn()
    except SystemExit:
        print(f"  PASS  {name}")
        return
    FAIL.append(name)
    print(f"  FAIL  {name}: expected SystemExit, none raised")


def write_trial(
    root: str,
    prefix: str,
    tid: str,
    *,
    correct: int,
    gold: str,
    verdict: str,
    reward: bool = True,
    grades: bool = True,
    suffix: str = "aaa",
    extra: dict | None = None,
) -> None:
    v = os.path.join(root, prefix, f"{tid}__{suffix}", "verifier")
    os.makedirs(v, exist_ok=True)
    if reward:
        json.dump(
            {
                "outcome_score": float(correct),
                "outcome_label_parsed": 1,
                "report_present": 1,
                **(extra or {}),
            },
            open(os.path.join(v, "reward.json"), "w"),
        )
    if grades:
        json.dump(
            {"ground_truth_label": gold, "agent_label": verdict},
            open(os.path.join(v, "grades_detail.json"), "w"),
        )


def main() -> int:
    root = tempfile.mkdtemp(prefix="pfb_test_")
    try:
        # ---- fixture: 4 tasks x 3 attempts, one system. Hand-computable.
        #   t1 YES  correct in all 3            -> pass^3 yes, pass@3 yes
        #   t2 YES  correct in attempts 1,2     -> pass^3 no,  pass@3 yes
        #   t3 NO   never correct               -> pass^3 no,  pass@3 no
        #   t4 NO   correct in attempts 2,3     -> pass^3 no,  pass@3 yes
        #
        # The per-attempt accuracies are deliberately UNEQUAL (50, 75, 50). An earlier fixture had them
        # all equal, which made avg@3 - pass^3 and pass@3 - avg@3 numerically identical -- so the
        # Overst. assertion passed under either formula and could not catch the inversion it exists to
        # catch. Any fixture for a difference-of-means metric has to break that symmetry.
        plan = {
            "t1-aaaaaa": ("YES", [1, 1, 1]),
            "t2-bbbbbb": ("YES", [1, 1, 0]),
            "t3-cccccc": ("NO", [0, 0, 0]),
            "t4-dddddd": ("NO", [0, 1, 1]),
        }
        for i, pfx in enumerate(["A", "B", "C"]):
            for tid, (gold, cs) in plan.items():
                write_trial(
                    root, pfx, tid, correct=cs[i], gold=gold, verdict=gold if cs[i] else "OTHER"
                )
        atts = [pfb.read_bundle_local(root, p) for p in ("A", "B", "C")]
        r = pfb.score_arm(atts)

        check("n_com", r["n"], 4)
        # per-attempt accuracy: A 2/4=50, B 3/4=75, C 2/4=50
        check("per-attempt acc", [round(x, 4) for x in r["per_acc"]], [50.0, 75.0, 50.0])
        check("avg@3", round(r["avg"], 4), 58.3333)
        check("SD across attempts", round(r["sd"], 3), 14.434)
        check("pass^3 (t1 only)", round(r["pow"], 4), 25.0)
        check("pass@3 (t1,t2,t4)", round(r["at"], 4), 75.0)
        check("spread = pass@3 - pass^3", round(r["spread"], 4), 50.0)
        # Overst. is avg@3 - pass^3, NOT pass@3 - avg@3. Inverting it left the per-class table correct
        # and one summary column wrong on every row, which is why it gets its own assertion.
        # avg@3 - pass^3 = 33.3333; the inverted pass@3 - avg@3 would be 16.6667 -- distinguishable now.
        check("Overst. = avg@3 - pass^3", round(r["overst"], 4), 33.3333)
        # verdict agreement: t1 identical (YES x3); t3 identical (OTHER x3); t2,t4 differ
        check("verdict agreement", round(r["agree"], 4), 50.0)
        # agreed-and-wrong: t3 only (t1 agreed and was correct in attempt 1)
        check("agreed-but-wrong", round(r["agree_wr"], 4), 25.0)

        pc = r["pc"]
        check("per-class YES n", pc["YES"]["n"], 2)
        check("per-class YES pass^3 (t1 of t1,t2)", round(pc["YES"]["pow"], 4), 50.0)
        check("per-class YES pass@3", round(pc["YES"]["at"], 4), 100.0)
        # YES accuracies by attempt: A 2/2=100, B 2/2=100, C 1/2=50 -> stdev([100,100,50]) = 28.8675
        check("per-class YES SD", round(pc["YES"]["sd"], 3), 28.868)
        check("per-class NO n", pc["NO"]["n"], 2)
        check("per-class NO pass^3", round(pc["NO"]["pow"], 4), 0.0)
        check("per-class NO pass@3 (t4)", round(pc["NO"]["at"], 4), 50.0)
        check("class supports sum to n_com", sum(v["n"] for v in pc.values()), r["n"])

        # ---- the intersection rule: a task absent from one attempt is dropped entirely, because
        # "correct on every attempt" is trivially true of the attempts that happen to exist.
        write_trial(root, "A", "t5-eeeeee", correct=1, gold="YES", verdict="YES")
        write_trial(root, "B", "t5-eeeeee", correct=1, gold="YES", verdict="YES")
        atts2 = [pfb.read_bundle_local(root, p) for p in ("A", "B", "C")]
        check("task in 2 of 3 attempts is excluded", pfb.score_arm(atts2)["n"], 4)

        # ---- a trial with no reward.json is not gradeable (same rule as score_clinicare)
        write_trial(root, "D", "t9-ffffff", correct=1, gold="YES", verdict="YES", reward=False)
        check("no reward.json -> not gradeable", len(pfb.read_bundle_local(root, "D")), 0)

        # ---- a reward written by a verifier that failed before scoring is not gradeable either: its
        # 0.0 is not a verdict, so counting it would score an unchecked trial as a wrong answer.
        for i, key in enumerate(VERIFIER_FAILED_KEYS):
            write_trial(
                root, f"V{i}", "t7-hhhhhh", correct=0, gold="YES", verdict="NO", extra={key: 1}
            )
            check(f"{key} -> not gradeable", len(pfb.read_bundle_local(root, f"V{i}")), 0)
        # The local copy (this script runs without the package installed) must match the rule.
        check(
            "VERIFIER_FAILED_KEYS matches clinicare_core",
            pfb.VERIFIER_FAILED_KEYS,
            VERIFIER_FAILED_KEYS,
        )

        # ---- reward.json WITHOUT grades_detail.json is also not gradeable. Admitting it counted the
        # trial in n_com while no per-class bucket could hold it, so the supports stopped summing to
        # n_com, and gold+verdict going missing together put every attempt on the same "(no verdict)"
        # sentinel, scoring them as agreeing. Both failures printed clean numbers.
        write_trial(root, "F", "t8-gggggg", correct=1, gold="YES", verdict="YES", grades=False)
        check(
            "reward.json without grades_detail -> not gradeable",
            len(pfb.read_bundle_local(root, "F")),
            0,
        )

        for pfx in ("G1", "G2", "G3"):
            write_trial(root, pfx, "keep-1", correct=1, gold="YES", verdict="YES")
            write_trial(root, pfx, "drop-1", correct=0, gold="YES", verdict="YES", grades=False)
        g = pfb.score_arm([pfb.read_bundle_local(root, p) for p in ("G1", "G2", "G3")])
        check("reward-only trial excluded from n_com", g["n"], 1)
        check(
            "class supports still sum to n_com with a reward-only trial present",
            sum(v["n"] for v in g["pc"].values()),
            g["n"],
        )

        # ---- but gold WITH no parsed verdict is a real outcome and must be KEPT, participating in the
        # agreement columns through the sentinel -- the reference implementation does the same.
        for pfx in ("H1", "H2", "H3"):
            write_trial(root, pfx, "hasgold-1", correct=0, gold="NO", verdict="")
        h = pfb.score_arm([pfb.read_bundle_local(root, p) for p in ("H1", "H2", "H3")])
        check("gold present but no verdict -> kept", h["n"], 1)
        check("its class support is counted", sum(v["n"] for v in h["pc"].values()), 1)

        # ---- refusals
        write_trial(root, "E", "t1-aaaaaa", correct=1, gold="YES", verdict="YES", suffix="one")
        write_trial(root, "E", "t1-aaaaaa", correct=0, gold="YES", verdict="NO", suffix="two")
        check_raises(
            "two graded dirs for one task id refuses", lambda: pfb.read_bundle_local(root, "E")
        )
        check_raises(
            "missing bundle refuses", lambda: pfb.read_bundle_local(root, "does-not-exist")
        )
        check_raises(
            "empty intersection refuses",
            lambda: pfb.score_arm(
                [
                    {"a": {"correct": 1, "gold": "YES", "verdict": "YES"}},
                    {"b": {"correct": 1, "gold": "YES", "verdict": "YES"}},
                ]
            ),
        )

        # ---- emit() shape: one row per system in each of the two tables, 16 per-class cells per row
        res = {system: r for _, system in pfb.ROWS}
        block = pfb.emit(res)
        rows = [l for l in block.splitlines() if "&" in l and not l.startswith("%")]
        check("emit row count (5 systems x 2 tables)", len(rows), 2 * len(pfb.ROWS))
        perclass = rows[len(pfb.ROWS) :]
        check("per-class row has 16 value cells", perclass[0].count("&"), 16)
        check(
            "class order is yes,no,lack,amb",
            pfb.CLASS_ORDER,
            ["YES", "NO", "INDETERMINATE_LACK_OF_DATA", "INDETERMINATE_MEDICAL_AMBIGUITY"],
        )

        # ---- a class absent from the data must render as -- rather than crash or print 0
        only_yes = [
            {f"t{i}": {"correct": 1, "gold": "YES", "verdict": "YES"} for i in range(3)}
        ] * 3
        rr = pfb.score_arm([dict(d) for d in only_yes])
        b2 = pfb.emit({system: rr for _, system in pfb.ROWS})
        check("absent class renders as --", "-- & -- & -- & --" in b2, True)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{len(FAIL)} failure(s)" if FAIL else "\nall checks passed")
    return 1 if FAIL else 0


def test_passk_from_bundles() -> None:
    """pytest entry point.

    Without this, pytest collects this file (it matches test_*.py) and finds no test functions, so CI
    reports a pass having run nothing -- a check that cannot fail. The assertions live in main() so the
    file also runs standalone without pytest installed, which is how it is run on this host.
    """
    assert main() == 0, "one or more checks failed; see captured stdout for which"


if __name__ == "__main__":
    raise SystemExit(main())
