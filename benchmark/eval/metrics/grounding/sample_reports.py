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

Paired design (the citation coverage / citation precision table): the same cases for every system.

  python benchmark/eval/metrics/grounding/sample_reports.py \\
      --traces <dir with full750-clinicare-* arms> --paired-per-scenario 2 --seed 20261125 \\
      --out-list paired_trials.txt --out-csv paired_manifest.csv
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
from core.trajectory import grounding_scoreable  # noqa: E402

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
    return grounding_scoreable(trial)


def paired(args, want: list[str], rng: random.Random) -> int:
    """The same cases for every system, so that between-system differences are within-case
    differences. A case is the trial-directory name up to `__`. Within each scenario the cases are
    shuffled with the seed and taken in order until `--paired-per-scenario` of them have a
    scoreable trial in every arm; cases on which any arm produced no gradeable report are skipped,
    so the sample leans toward cases no system failed on."""
    runs = sorted(p for p in args.traces.iterdir() if p.is_dir() and family_of(p.name, args.cohort) in want)
    if not runs:
        print("  ! no arms found", file=sys.stderr)
        return 2
    trials: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    scenario: dict[str, str] = {}
    for run_dir in runs:
        for trial in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            slug = slug_from_trial_dir(trial.name)
            if slug:
                case = trial.name.split("__")[0]
                trials[case][run_dir.name].append(trial)
                scenario[case] = slug
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for case, slug in scenario.items():
        by_scenario[slug].append(case)
    chosen: list[tuple[str, str, dict[str, Path]]] = []
    for slug in sorted(by_scenario):
        cases = sorted(by_scenario[slug])
        rng.shuffle(cases)
        got = tried = 0
        for case in cases:
            if got >= args.paired_per_scenario:
                break
            tried += 1
            picks: dict[str, Path] = {}
            for run_dir in runs:
                ok = [t for t in trials[case].get(run_dir.name, []) if scoreable(t)]
                if not ok:
                    break
                picks[run_dir.name] = ok[0]
            if len(picks) == len(runs):
                chosen.append((slug, case, picks))
                got += 1
        print(f"  {slug}: {got} cases (tried {tried})")
    args.out_list.write_text("\n".join(str(t.resolve()) for _, _, picks in chosen for t in picks.values()) + "\n")
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "run", "scenario", "case", "trial"])
        for slug, case, picks in chosen:
            for run, t in picks.items():
                w.writerow([family_of(run, args.cohort), run, slug, case, t.name])
    n = sum(len(picks) for _, _, picks in chosen)
    print(f"wrote {len(chosen)} cases x {len(runs)} arms = {n} trials → {args.out_list}, manifest → {args.out_csv}")
    return 0 if chosen else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--per-family", type=int, default=50)
    ap.add_argument("--families", default="claude-code,codex,gemini-cli,opencode")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--cohort", default="full750", help="arm-name prefix to sample from (default: the paper's full750 arms)")
    ap.add_argument("--paired-per-scenario", type=int, default=0,
                    help="paired design: pick this many CASES per scenario for which every arm of the wanted "
                         "families has a scoreable trial, and list that case's trial from every arm "
                         "(e.g. 2 gives 50 cases x 16 systems). Overrides --per-family.")
    ap.add_argument("--out-list", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()
    want = [f.strip() for f in args.families.split(",") if f.strip()]
    rng = random.Random(args.seed)
    if args.paired_per_scenario:
        return paired(args, want, rng)

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
