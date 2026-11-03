"""sample_reports — pick a stratified sample of grounding-scoreable trials for the pool-vs-cited
P_claim comparison and the hand-reviewed citation sample.

Why a sample: cited mode costs a judge call per distinct resolved evidence set per report, and the
question it answers first — does judging against the cited record instead of the whole pool change
the picture? — is answerable on a few hundred reports. The sample is stratified by harness family
(the axis Fig. 6 is about), and within a family spread evenly over models and scenarios.

Eligibility matches the paper's grounding-scoreable definition: a report is present, the agent
retrieved at least one patient record, and the report does not declare the environment degraded.
"Retrieved a patient record" means a MIMIC data tool returned rows; in code-mode traces (exec
scaffold, bash) it means a recovered MIMIC call plus non-empty command output, since the rows live
in the output. Sub-agent summaries and command output with no MIMIC call do not qualify.

  python benchmark/eval/metrics/grounding/sample_reports.py \\
      --traces <dir with full750-clinicare-* arms> --per-family 50 --seed 20261102 \\
      --out-list sample_trials.txt --out-csv sample_manifest.csv

`--out-list` is consumed by insight_grounding_llm.py `--trials-from`.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from clinicare_core.paths import bootstrap

bootstrap(__file__, "eval")
from core.scenarios import slug_from_trial_dir  # noqa: E402
from core.trajectory import EXEC_OUTPUT_TOOL, MIMIC_TOOLS, load_trajectory  # noqa: E402

# Harness family from the arm name (run_sweep.py tags): the harness is the first token after
# the cohort prefix. `claude` is Claude Code, `codex` is Codex, `gemini` is Gemini CLI,
# `opencode` the neutral harness; the fixed-model ablation arms carry `qwen` / `miniswe`.
FAMILY = {"claude": "claude-code", "codex": "codex", "gemini": "gemini-cli", "opencode": "opencode",
          "qwen": "qwen-coder", "miniswe": "mini-swe-agent"}


# Arms that are not among the paper's 16 systems: the MedHELM additions and the Nemotron run
# postdate the submission, and the harness-* arms are the fixed-model ablation.
NOT_IN_PAPER = ("gpt6astra", "gemini-38flash", "nemotronlightning")


def family_of(run: str, cohort: str = "full750") -> str | None:
    parts = run.split("-")
    # <cohort>-clinicare-<harness>-<model>-v3
    if len(parts) < 4 or parts[0] != cohort or parts[1] != "clinicare":
        return None
    if any(x in run for x in NOT_IN_PAPER):
        return None
    return FAMILY.get(parts[2])


def scoreable(trial: Path) -> bool:
    report = trial / "verifier" / "report.md"
    tj = trial / "agent" / "trajectory.json"
    if not report.exists() or not tj.exists():
        return False
    if "Environment: degraded" in report.read_text(errors="replace"):
        return False
    traj = load_trajectory(tj)
    obs = traj.observations
    if any(o.tool in MIMIC_TOOLS and o.text.strip() for o in obs):
        return True
    return any(o.tool in MIMIC_TOOLS for o in obs) and any(
        o.tool == EXEC_OUTPUT_TOOL and o.text.strip() for o in obs
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--per-family", type=int, default=50)
    ap.add_argument("--families", default="claude-code,codex,gemini-cli,opencode")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--cohort", default="full750", help="arm-name prefix to sample from (default: the paper's full750 arms)")
    ap.add_argument("--out-list", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()
    want = [f.strip() for f in args.families.split(",") if f.strip()]
    rng = random.Random(args.seed)

    # family -> model(run) -> scenario -> [trial dirs], eligibility checked lazily per pick.
    pool: dict[str, dict[str, dict[str, list[Path]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run_dir in sorted(p for p in args.traces.iterdir() if p.is_dir()):
        fam = family_of(run_dir.name, args.cohort)
        if fam not in want:
            continue
        for trial in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            slug = slug_from_trial_dir(trial.name)
            if slug:
                pool[fam][run_dir.name][slug].append(trial)

    chosen: list[tuple[str, str, str, Path]] = []
    for fam in want:
        runs = sorted(pool[fam])
        if not runs:
            print(f"  ! no arms for family {fam}", file=sys.stderr)
            continue
        # Round-robin over models, and within a model over scenarios in a seeded order, so 50
        # picks cover every model of the family and as many scenarios as possible.
        order = {r: rng.sample(sorted(pool[fam][r]), len(pool[fam][r])) for r in runs}
        for r in runs:
            for s in order[r]:
                rng.shuffle(pool[fam][r][s])
        picked = 0; cursor = {r: 0 for r in runs}; exhausted: set[str] = set()
        while picked < args.per_family and len(exhausted) < len(runs):
            for r in runs:
                if picked >= args.per_family or r in exhausted:
                    continue
                while cursor[r] < len(order[r]):
                    s = order[r][cursor[r]]; cursor[r] += 1
                    cands = [t for t in pool[fam][r][s] if scoreable(t)]
                    if cands:
                        chosen.append((fam, r, s, cands[0])); picked += 1
                        break
                else:
                    exhausted.add(r)
        print(f"  {fam}: {picked} trials from {len(runs)} arms")

    args.out_list.write_text("\n".join(str(t.resolve()) for _, _, _, t in chosen) + "\n")
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "run", "scenario", "trial"])
        for fam, r, s, t in chosen:
            w.writerow([fam, r, s, t.name])
    print(f"wrote {len(chosen)} trials → {args.out_list}, manifest → {args.out_csv}")
    return 0 if chosen else 2


if __name__ == "__main__":
    raise SystemExit(main())
