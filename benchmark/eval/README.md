# eval/ — everything that computes a metric

Every metric is computed here; `scripts/` is cohort and task **generation**. One thing sits
outside: the **in-trial verifier**, in [`../_template/tests/`](../_template/tests/) (`test.sh`,
`judge.py`, `policy_scoring.py`, `evidence.py`). `build_tasks.py` hardlinks those into every
generated task dir, so they must be standalone, import-free, and runnable in a container with no
repo checkout. The dependency therefore runs one way: `eval/` imports them, never the reverse.
That is why `evidence.py` owns `normalize_tool` / `_MCP_PREFIXES` and `eval/` reads them from
there rather than keeping a second copy, and why `policy_scoring.py` — imported by both surfaces
— is what keeps the in-trial and post-hoc policy numbers from drifting apart.

See the [benchmark README](../README.md#how-scoring-splits) for which metric belongs on which
surface and why.

## Layout

```
eval/
├── grade.py                  run every scorer below over a finished job dir, in one command
├── core/
│   └── trajectory.py         harness-agnostic trajectory loader + MIMIC observation extractor
├── metrics/                  one module per metric; each is a CLI over a job dir
│   ├── process.py            process rubric, post-hoc 
│   ├── policy_support.py     policy citation support + outcome re-grade  (step 5)
│   └── grounding/            patient-evidence grounding (see grounding/README.md)
│       ├── evidence_retrieval_recall.py   retrieval coverage — gold tables ∩ tables read
│       ├── insight_grounding.py           T1: cited record IDs vs retrieved evidence (no LLM)
│       ├── insight_grounding_llm.py       T2: claim-level supported/contradicted/unverifiable
│       ├── insights_recall.py             finding coverage vs required-evidence categories
│       └── policy_grounding.py            offline policy grounding over trace dumps
├── gold/                     derived gold sets + the builders that produce them
│   ├── policy_gold.json          per-scenario governing policy docs   ← build_tasks.py reads this
│   ├── required_evidence.json    per-scenario required-evidence categories
│   ├── gold_sources.json         per-scenario required MIMIC source tables
│   └── build_*.py                the builders (rerun only when the annotation changes)
├── reliability/              judge-agreement studies (Fleiss' κ, rep-noise, family bias)
│   ├── judge_panel.py / judge_analysis.py                 process rubric
│   └── policy_judge_panel.py / policy_judge_analysis.py   policy rubric
├── report/                   cross-trial aggregation → the tables you paste
│   ├── score_clinicare.py            models × metrics table (accuracy, F1, cost, efficiency)
│   ├── passk_from_bundles.py         repeated-run reliability (pass^k / pass@k) LaTeX blocks
│   ├── paired_tests.py               pairwise McNemar, tiers, re-ranking, instrument reliability
│   └── policy_grounding_summary.py   policy sweep summary + corpus-gap analysis
├── tests/                    test suites for the eval code (run them directly; no pytest needed)
│   ├── test_verdict.py           the PRIMARY reward: verdict extraction + 4-class match
│   ├── test_evidence.py          trajectory→evidence.jsonl extractor, per harness dialect
│   ├── test_policy_grounding.py  deterministic policy scorer + support-result semantics
│   ├── test_passk_from_bundles.py   pass^k / pass@k metric math + the refusals
│   └── test_paired_tests.py         McNemar / BH / Holm / alpha math + denominator rules
└── traces/      (gitignored) downloaded trace dumps
```

## Conventions

Every scorer under `metrics/` follows the same contract, so they compose and can't quietly
under-report:

* **Coverage gate.** Enumerates *every* trial dir in a job, reports `n_scored / n_trials` plus
  skip reasons, and exits non-zero only on *fixable* failures (unreadable rubrics, judge error).
  Legitimately unscorable trials — the agent wrote no report, the scenario has no rubric, policy-OFF —
  are reported separately and do not fail the run. `--limit` prints a PARTIAL banner and suppresses
  the gate, so a capped run can never be mistaken for a full one.
* **`--dry-run`** classifies and reports coverage without calling an LLM. Use it first; it's free.
* **Idempotent per-trial output** (`verifier/process.json`, `verifier/policy_support.json`), so a
  re-run resumes rather than re-billing. A cached payload carrying `error` is retried, not kept.
* **`--diff`** compares against per-trial numbers recorded IN the trial. The verifier is
  deterministic-only and does not record the LLM-judged ones, so this reports nothing on trials
  built from this tree; it applies to older trials that were graded in-container.
* **`scorer_version`** stamped into every output, so a figure traces to the code that made it.

## Quick start

```bash
cd <repo root>
export JUDGE_BASE_URL=... JUDGE_API_KEY=...        # from.env; JUDGE_MODEL there may be stale

HCE=benchmark/eval
python $HCE/metrics/process.py jobs/<job> --dry-run          # coverage only, no cost
python $HCE/metrics/process.py jobs/<job>                    # grade
python $HCE/metrics/policy_support.py jobs/<job>             # policy citation support
python $HCE/report/score_clinicare.py jobs/<job> --csv out.csv
python $HCE/report/paired_tests.py jobs/<job>                # can system A be ranked above B?
python $HCE/tests/test_verdict.py                            # primary-reward contract
python $HCE/tests/test_policy_grounding.py                   # deterministic policy tests
python $HCE/tests/test_evidence.py                           # evidence extractor tests
python $HCE/tests/test_paired_tests.py                       # paired-statistics tests
python $HCE/tests/test_grounding_scope.py                    # R_retrieval scope + bash code-mode parsing
python $HCE/tests/test_insight_grounding_cited.py            # which evidence reaches the P_claim judge
```

Grounding metrics run over a directory of trace dumps — see **Grounding metrics** below.

## Grounding metrics

Did the agent's report rest on evidence it actually retrieved? Scored offline from persisted
trial artifacts — no live MIMIC, no re-running the agent. The paper covers what these measure
and what they showed; this file is the runbook.

| metric | question | needs |
| :-- | :-- | :-- |
| `evidence_retrieval_recall` | did the agent read the source tables the answer requires? | committed gold, no LLM |
| `insight_grounding` (T1) | are the record IDs it cited present in what it retrieved? | trace only, no gold, no LLM |
| `insight_grounding_llm` (T2) | are its numeric / free-text claims supported by that evidence (`--evidence pool`), or by the records they cite (`--evidence cited`)? | LLM judge |
| `insights_recall` | did the report cover each required-evidence category? | committed rubric + LLM judge |

T1 is a **fabrication floor**, not a discriminator: an ID cannot be computed, so an ID absent
from every tool result was invented. It reads 1.00 on well-behaved runs by design. Two atom types
are deliberately excluded from it — timestamps (reports legitimately state *computed* window
boundaries that are never a database row) and bare decimals (a lab value like `0.9`
substring-matches by chance). Those are T2's job, which is where the discrimination lives.

### Running them

**Normally you don't run these individually — use `grade.py`, which runs every offline scorer
over a finished job dir in one command:**

```bash
export JUDGE_BASE_URL=... JUDGE_API_KEY=...
python benchmark/eval/grade.py jobs/<job>              # everything
python benchmark/eval/grade.py jobs/<job> --dry-run    # coverage only, spends nothing
python benchmark/eval/grade.py jobs/<job> --skip-llm   # deterministic stages only
```

It reports each stage as ran / skipped-with-reason / failed, and exits non-zero if a stage that
should have run failed. Two things it handles that are easy to get wrong by hand: the grounding
metrics take a dir *containing* job dirs (`--traces`), not a job dir, so run them straight at
`jobs/` and you score every job you have ever run — `grade.py` scopes them to the jobs you asked
for; and `process`/`policy_support` cache per trial while the LLM grounding metrics do not, so on
a large sweep the latter dominate cost and wall-clock (use `--limit` to sample them).

The rest of this section is the manual invocation, for when you want one metric in isolation.
It needs `JUDGE_BASE_URL` / `JUDGE_API_KEY` for the LLM tiers (see `.env.template`) and a
directory of traces — one dir per trial, each holding `agent/trajectory.json` and
`verifier/report.md`, which is exactly the layout `harbor run` writes under `jobs/<job>/`.

Steps 2 and 6 rebuild committed gold and are **only** needed if that gold changes. Normally you
skip both.

```bash
cd <repo root>
G=benchmark/eval/metrics/grounding

# 2. (only if the gold changes) rebuild gold_sources.json from the clinician annotation.
#    Requires your own copy: the delivery is a MIMIC derivative and is not publicly distributed.
python benchmark/eval/gold/build_gold.py --annotation <your-copy.csv>

# 3. evidence_retrieval_recall — reads the committed gold_sources.json, no LLM.
#    A required table is credited only when the call that read it named the case's patient
#    (subject_id in the typed args or in the run_sql text); calls that name no subject_id are
#    credited and counted `unscoped`; another encounter of the same patient is counted, not
#    penalised. --no-scope reproduces the published numbers; --out writes per-trial JSON.
python $G/evidence_retrieval_recall.py --traces <traces> [--no-scope] [--out rr.json]

# 4. insight_grounding T1 — trace-only, no gold, no LLM
python $G/insight_grounding.py --traces <traces>

# 5. insight_grounding T2 — LLM judge (start with --limit to sanity-check cost)
python $G/insight_grounding_llm.py --traces <traces> --limit 3 --verbose
#    --evidence pool  (default) judges every claim against everything the agent retrieved —
#                     retrieved-evidence support, the published P_claim; evidence_refs="all"
#    --evidence cited judges each claim only against the records its citation resolves to
#                     (claim–citation pairs); adds unresolvable / unresolved verdicts,
#                     resolvable_frac and strict_on_resolvable; separate cache file
#    --trials-from F  grade only the trial dirs listed in F (see sample_reports.py)
python $G/insight_grounding_llm.py --traces <traces> --evidence cited --trials-from sample.txt

# 6. (only if the annotation changes) rebuild the required-evidence rubric
python benchmark/eval/gold/build_required_evidence.py --annotation <your-copy.csv>

# 7. insights_recall — category coverage vs the committed required_evidence.json
python $G/insights_recall.py --traces <traces>
```

`gold_sources.json` and `required_evidence.json` are scenario-level derived rubrics with no
patient identifiers, which is why they are committed even though the clinician annotation they
are distilled from is not distributable. Shipping them is also what makes the numbers
reproducible: `required_evidence.json` is LLM-distilled, so regenerating it would move
$R_{\mathrm{finding}}$ and invalidate any comparison against the published values.

### Reading `evidence_retrieval_recall` with care

It weights every gold table equally, so a low score conflates genuine under-retrieval with
skipping non-decisive *context* tables the annotation happened to list. On `advanced-imaging` the
answer hinges on `poe` + `icustays` + `radiology`, which agents do read, yet demographics tables
in the gold pull recall to ~0.60. Treat it as a coarse signal and prefer `insights_recall`
(scored against the expert's decisive-evidence rubric) where the two disagree.

Precision is deliberately not reported: coarse tools like `get_admission_details` read six tables
per call, so "tables retrieved" is over-counted and precision would not mean anything.
