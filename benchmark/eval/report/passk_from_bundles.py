"""pass^k / pass@k reliability tables for the CliniCARE-Bench development subset.

Emits the two LaTeX row blocks the paper uses: the per-system summary (avg@3, SD, pass^3, pass@3, spread,
overstatement, verdict agreement) and the per-class breakdown by gold verdict class.

Run this over YOUR OWN repeated runs: one job dir per (system, attempt), all from a single run
generation. Graded outputs are not redistributed, so there is no bundle to download — see the
artifact table in the top-level README for why. Each job dir must hold exactly one graded trial
per task id, with the reference verdict already reconciled into `verifier/reward.json`; that is
what lets this skip deduplication, gold correction, and any build directory. Everything needed is
two files per trial:

    verifier/reward.json         outcome_score        1.0 == the verdict was correct
    verifier/grades_detail.json  ground_truth_label   the gold class
                                 agent_label          the emitted verdict (for the agreement columns)

Attempts map to prefixes: attempt 1 = dev148-*, attempt 2 = passk2-*, attempt 3 = passk3-*.

Definitions (k = 3 here):
    avg@k     mean of the per-attempt accuracies, all over the same denominator
    pass^k    correct on EVERY attempt -- the deployment floor
    pass@k    correct on AT LEAST ONE attempt -- the oracle-verifier ceiling
    spread    pass@k - pass^k; 0 means deterministic on these tasks
    Overst.   avg@k - pass^k; how far a single-attempt score overstates the floor
    Agree     identical verdict on all k attempts, correct or not
    Agree-wr  agreed on all k AND wrong -- consistently wrong, not flaky

Only tasks graded in ALL k attempts are eligible. A task missing from one attempt would otherwise
inflate pass^k, since "correct on every attempt" is trivially true of the attempts that exist.

Usage:
    python3 benchmark/eval/report/passk_from_bundles.py --root /path/to/bundles

    # regression-check the numbers against a known-good LaTeX block
    python3 benchmark/eval/report/passk_from_bundles.py --root <bundles> --validate <ref.tex>

Tests: benchmark/eval/tests/test_passk_from_bundles.py (no bundles required).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

# reward.json markers of a verifier that failed before scoring (its 0.0 is not a verdict). A copy of
# clinicare_core.rewards.VERIFIER_FAILED_KEYS, kept local because this script runs under a bare
# `python3` with no package installed; tests/test_passk_from_bundles.py asserts the two are equal.
VERIFIER_FAILED_KEYS = ("rubrics_load_failed", "judge_launch_failed")

# Display name and system slug for the REPEATED-RUN RELIABILITY tables, in the order the paper
# prints them. These five are the systems run with k=3 on the development subset — NOT the 16-system
# leaderboard, which is a different (larger) set. Row order is data, not presentation: reordering
# here silently reorders the table, so it is stated once and asserted in the tests rather than
# derived from whatever order the bundles are read in.
ROWS = [
    ("Opus 4.8 (Claude Code)", "claude-opus48"),
    ("GPT-5.5 (Codex)", "codex-gpt55"),
    ("GPT-5.6-Sol (Codex)", "codex-gpt56sol"),
    ("GPT-5.6-Luna (Codex)", "codex-gpt56luna"),
    ("Gemini-3.1-Pro (Gemini CLI)", "gemini-31pro"),
]
ATTEMPT_PREFIX = ["dev148", "passk2", "passk3"]
CLASS_ORDER = ["YES", "NO", "INDETERMINATE_LACK_OF_DATA", "INDETERMINATE_MEDICAL_AMBIGUITY"]


def _prefix(attempt_ix: int, system: str) -> str:
    return f"{ATTEMPT_PREFIX[attempt_ix]}-clinicare-{system}-v3"


def read_bundle_local(root: str, prefix: str) -> dict[str, dict]:
    """-> {task_id: {correct, gold, verdict}} for one bundle directory.

    A trial dir is `<task_id>__<hash>`. A bundle holds exactly one per task id, so no dedup: if a
    second dir for the same id ever appears, that is a defect in the bundle and we refuse rather than
    silently pick one -- picking would make the number depend on directory order.
    """
    d = os.path.join(root, prefix)
    if not os.path.isdir(d):
        raise SystemExit(f"REFUSING: no bundle at {d}")
    out: dict[str, dict] = {}
    for e in os.scandir(d):
        if not e.is_dir() or e.name.startswith("."):
            continue
        tid = e.name.split("__")[0]
        rec = _read_trial(os.path.join(e.path, "verifier"))
        if rec is None:
            continue
        if tid in out:
            raise SystemExit(
                f"REFUSING: {prefix} has two graded dirs for {tid}; each attempt's job dir "
                f"must hold exactly one trial per task id"
            )
        out[tid] = rec
    return out


def _read_trial(vdir: str) -> dict | None:
    """None when the trial is not gradeable -- no reward.json, or one written by a verifier that failed
    before scoring (clinicare_core.rewards) -- the same rule score_clinicare uses."""
    rp, gp = os.path.join(vdir, "reward.json"), os.path.join(vdir, "grades_detail.json")
    if not os.path.exists(rp):
        return None
    try:
        rj = json.load(open(rp))
    except (OSError, json.JSONDecodeError):
        return None
    if any(rj.get(key) == 1 for key in VERIFIER_FAILED_KEYS):
        return None  # the verifier failed before scoring: its 0.0 is not a verdict
    gold = verdict = None
    if os.path.exists(gp):
        try:
            gj = json.load(open(gp)) or {}
            gold = (gj.get("ground_truth_label") or "").strip() or None
            verdict = (gj.get("agent_label") or "").strip() or None
        except (OSError, json.JSONDecodeError):
            pass
    # A trial with no gold class cannot be scored here, so it is NOT gradeable -- the same treatment a
    # missing reward.json gets. Admitting it instead put it in n_com while no per-class bucket could
    # hold it, so the four class supports no longer summed to n_com and the two emitted tables
    # contradicted each other; and because gold and verdict go missing together, all such attempts
    # collapsed onto the same "(no verdict)" sentinel and were counted as AGREEING on a verdict none of
    # them produced. Both were silent: every number still printed.
    #
    # `verdict` alone missing is different and is kept: a graded trial that emitted no parsable verdict
    # is a real outcome, and the sentinel is how it participates in the agreement columns.
    if gold is None:
        return None
    sc = rj.get("outcome_score")
    if sc is None:
        sc = rj.get("score")
    return {"correct": int(float(sc or 0) == 1.0), "gold": gold, "verdict": verdict}


def score_arm(attempts: list[dict[str, dict]]) -> dict:
    """pass^k / pass@k / agreement over the tasks gradeable in EVERY attempt.

    The intersection is the point: a task missing from one attempt would otherwise inflate pass^k,
    because "correct on every attempt" is trivially satisfied by the attempts that exist.
    """
    elig = sorted(set.intersection(*(set(a) for a in attempts)))
    if not elig:
        raise SystemExit("REFUSING: no task id is graded in all three attempts")
    n = len(elig)
    per_acc = [100 * sum(a[t]["correct"] for t in elig) / n for a in attempts]
    pow_ = 100 * sum(1 for t in elig if all(a[t]["correct"] for a in attempts)) / n
    at = 100 * sum(1 for t in elig if any(a[t]["correct"] for a in attempts)) / n
    agree = (
        100
        * sum(1 for t in elig if len({(a[t]["verdict"] or "(no verdict)") for a in attempts}) == 1)
        / n
    )
    agree_wr = (
        100
        * sum(
            1
            for t in elig
            if len({(a[t]["verdict"] or "(no verdict)") for a in attempts}) == 1
            and not attempts[0][t]["correct"]
        )
        / n
    )
    pc = {}
    for c in {attempts[0][t]["gold"] for t in elig if attempts[0][t]["gold"]}:
        ts = [t for t in elig if attempts[0][t]["gold"] == c]
        accs = [100 * sum(a[t]["correct"] for t in ts) / len(ts) for a in attempts]
        pc[c] = {
            "n": len(ts),
            "pow": 100 * sum(1 for t in ts if all(a[t]["correct"] for a in attempts)) / len(ts),
            "at": 100 * sum(1 for t in ts if any(a[t]["correct"] for a in attempts)) / len(ts),
            "sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        }
    return {
        "n": n,
        "per_acc": per_acc,
        "avg": sum(per_acc) / len(per_acc),
        "sd": statistics.stdev(per_acc) if len(per_acc) > 1 else 0.0,
        "pow": pow_,
        "at": at,
        "spread": at - pow_,
        # avg@3 - pass^3: how far a single-attempt score overstates the every-attempt floor.
        # NOT pass@3 - avg@3; that is a different quantity and gets the sign of the gap wrong.
        "overst": sum(per_acc) / len(per_acc) - pow_,
        "agree": agree,
        "agree_wr": agree_wr,
        "pc": pc,
    }


def _f(v: float, w: int = 4) -> str:
    """One decimal, \\phantom-padded so columns align -- the convention the existing block uses."""
    s = f"{v:.1f}"
    return "\\phantom{0}" * max(0, w - len(s)) + s


def emit(res: dict[str, dict]) -> str:
    L = [
        "% GENERATED by benchmark/eval/report/passk_from_bundles.py -- do not hand-edit.",
        "% Reproduces tab:passk / tab:passk-perclass from graded trials, with no task build needed.",
        "%",
        "% ---------- tab:passk",
    ]
    for name, system in ROWS:
        r = res[system]
        L.append(
            f"{name:31} & {r['n']} & {r['avg']:.1f} & {r['sd']:.1f} & {r['pow']:.1f} & "
            f"{r['at']:.1f} & {r['spread']:.1f} & {_f(r['overst'])} & {r['agree']:.1f} & "
            f"{r['agree_wr']:.1f} \\\\"
        )
    L += ["%", "% ---------- tab:passk-perclass"]
    for name, system in ROWS:
        pc = res[system]["pc"]
        cells = []
        for c in CLASS_ORDER:
            d = pc.get(c)
            cells += (
                ["--", "--", "--", "--"]
                if d is None
                else [str(d["n"]), f"{d['pow']:.1f}", f"{d['at']:.1f}", f"{d['sd']:.1f}"]
            )
        L.append(f"{name:31} & " + " & ".join(cells) + " \\\\")
    return "\n".join(L) + "\n"


_NUM = re.compile(r"-?\d+\.?\d*")


def validate(mine: str, ref_path: str) -> int:
    """Compare NUMBERS ONLY against a reference block, keyed by row label.

    Numbers rather than text because the two generators legitimately differ in provenance comments,
    padding and bold markup; a textual diff would report those as failures and bury a real one.
    """
    if not os.path.exists(ref_path):
        print(f"REFUSING: no reference at {ref_path}")
        return 2

    def rows(text: str) -> dict[str, list[str]]:
        out = {}
        for ln in text.splitlines():
            if "&" not in ln or ln.lstrip().startswith("%"):
                continue
            label = ln.split("&")[0].strip()
            if not any(label.startswith(n.split(" (")[0]) for n, _ in ROWS):
                continue
            nums = _NUM.findall(ln.split("&", 1)[1])
            out.setdefault(label, []).append(",".join(nums))
        return out

    a, b = rows(mine), rows(open(ref_path).read())
    bad = 0
    for label in sorted(set(a) | set(b)):
        xa, xb = a.get(label, []), b.get(label, [])
        if xa == xb:
            print(f"  MATCH {label}")
        else:
            bad += 1
            print(f"  DIFF  {label}")
            for i, (p, q) in enumerate(zip(xa + [""] * len(xb), xb + [""] * len(xa))):
                if p != q:
                    print(f"          block {i}: mine={p}\n          block {i}:  ref={q}")
    print(f"\n{'ALL ROWS MATCH' if not bad else f'{bad} ROW(S) DIFFER'}")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True, help="directory holding the bundles")
    ap.add_argument("--out", default="", help="write the LaTeX block here (default: stdout)")
    ap.add_argument(
        "--validate",
        default="",
        help="regression check: compare every number against this known-good .tex "
        "row block and exit non-zero on any difference",
    )
    a = ap.parse_args()

    res = {}
    for name, system in ROWS:
        atts = []
        for i in range(3):
            pfx = _prefix(i, system)
            atts.append(read_bundle_local(a.root, pfx))
        res[system] = score_arm(atts)
        r = res[system]
        print(
            f"  {system:18} n_com={r['n']:3}  avg@3={r['avg']:5.1f}  pass^3={r['pow']:5.1f}  "
            f"pass@3={r['at']:5.1f}",
            file=sys.stderr,
        )

    ns = {res[system]["n"] for _, system in ROWS}
    if len(ns) != 1:
        print(
            f"NOTE: n_com differs across systems {sorted(ns)}; the published caption asserts one n.",
            file=sys.stderr,
        )

    block = emit(res)
    if a.validate:
        return validate(block, a.validate)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(block)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
