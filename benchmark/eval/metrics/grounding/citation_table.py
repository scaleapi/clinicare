"""citation_table — per-system citation coverage and citation precision on a paired sample.

Reads the per-trial cited-mode results written by `insight_grounding_llm.py --evidence cited` for
the trials in a paired manifest (sample_reports.py --paired-per-scenario) and prints one row per
system: coverage and precision with percentile-bootstrap intervals over cases, the sentence-only
reading of both, and the share of claims whose only citation is a date. No judge calls.

Every system is averaged over the same cases. A case without a usable result for some system is
left out for all of them and listed, and the exit status is 1.

  python benchmark/eval/metrics/grounding/citation_table.py \\
      --traces <dir with the arms> --manifest paired_manifest.csv [--out table.json]

coverage  = claims that cite a record id, a tool or table, or a date / all claims      (per report)
precision = cited claims the judge found supported from the cited evidence alone / cited claims
A report with no cited claim has coverage 0 and contributes no precision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

CACHE = "insight_grounding_llm_cited.json"


def report_stats(d: dict) -> dict:
    det = d["detail"]
    cited = [c for c in det if c.get("citation_kind") == "pointer"]
    sent = [c for c in cited if c.get("cited_scope") == "sentence"]
    sup = lambda cs: sum(1 for c in cs if c.get("verdict") == "supported")  # noqa: E731
    return {
        "n": len(det),
        "coverage": len(cited) / len(det),
        "precision": sup(cited) / len(cited) if cited else None,
        "coverage_sentence": len(sent) / len(det),
        "precision_sentence": sup(sent) / len(sent) if sent else None,
        "date_only": sum(1 for c in cited if c.get("pointer") == "date") / len(det),
    }


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(vals: list[float | None], n_boot: int, seed: int) -> tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    bs = sorted(mean([rng.choice(vals) for _ in vals]) for _ in range(n_boot))
    return bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # One usable result per (system, case). A comparison between systems is only paired when
    # every system is averaged over the same cases, so a case that any system lacks (a grading
    # run that stopped early, a judge error) is left out for all of them and named.
    results: dict[str, dict[str, dict]] = defaultdict(dict)
    family: dict[str, str] = {}
    cases: list[str] = []
    for m in csv.DictReader(args.manifest.open()):
        family[m["run"]] = m["family"]
        if m["case"] not in cases:
            cases.append(m["case"])
        f = args.traces / m["run"] / m["trial"] / "verifier" / CACHE
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if d.get("error") or not d.get("n_claims"):
            continue
        results[m["run"]][m["case"]] = report_stats(d)
    if not results:
        print("no cited-mode results found for the manifest", file=sys.stderr)
        return 2
    complete = [c for c in cases if all(c in results[run] for run in family)]
    dropped = [c for c in cases if c not in complete]
    if dropped:
        lacking = {run: sum(c not in results[run] for c in cases) for run in family}
        print(f"  ! {len(dropped)} of {len(cases)} cases lack a usable {CACHE} for at least one system and are "
              f"left out for every system: {', '.join(dropped)}", file=sys.stderr)
        print("    missing per system: " + ", ".join(f"{r} {n}" for r, n in sorted(lacking.items()) if n), file=sys.stderr)
    if not complete:
        print("no case has a usable result for every system", file=sys.stderr)
        return 2
    per_run = {run: [results[run][c] for c in complete] for run in family}
    print(f"cases scored for every system: {len(complete)} of {len(cases)}")

    rows = []
    print(f"{'family':12s} {'run':40s} {'n':>3s} {'coverage':>8s} {'95% CI':>15s} {'precision':>9s} {'95% CI':>15s} "
          f"{'cov(s)':>6s} {'prec(s)':>7s} {'date':>5s}")
    for run in sorted(per_run, key=lambda r: (family[r], r)):
        rs = per_run[run]
        cov, prec = mean([r["coverage"] for r in rs]), mean([r["precision"] for r in rs])
        c_ci = boot_ci([r["coverage"] for r in rs], args.n_boot, args.seed)
        p_ci = boot_ci([r["precision"] for r in rs], args.n_boot, args.seed)
        row = {"family": family[run], "run": run, "n": len(rs), "claims": sum(r["n"] for r in rs),
               "coverage": cov, "coverage_ci": c_ci, "precision": prec, "precision_ci": p_ci,
               "coverage_sentence": mean([r["coverage_sentence"] for r in rs]),
               "precision_sentence": mean([r["precision_sentence"] for r in rs]),
               "date_only": mean([r["date_only"] for r in rs])}
        rows.append(row)
        print(f"{row['family']:12s} {run[:40]:40s} {row['n']:3d} {cov:8.3f} [{c_ci[0]:.3f}, {c_ci[1]:.3f}] {prec:9.3f} "
              f"[{p_ci[0]:.3f}, {p_ci[1]:.3f}] {row['coverage_sentence']:6.3f} {row['precision_sentence']:7.3f} {row['date_only']:5.2f}")
    print("\nfamily means (mean of system means):")
    for fam in sorted(set(family.values())):
        fr = [r for r in rows if r["family"] == fam]
        print(f"  {fam:12s} coverage {mean([r['coverage'] for r in fr]):.3f}   precision {mean([r['precision'] for r in fr]):.3f}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 1 if dropped else 0   # a table over fewer cases than the manifest is not the full result


if __name__ == "__main__":
    raise SystemExit(main())
