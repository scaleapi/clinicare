# benchmark/ — running CliniCARE

Everything needed to build the tasks, run agents against them, and score the results. If you
want to know how the benchmark was *constructed* — cohort selection, label adjudication, corpus
ingestion — that's [`../construction/`](../construction/), and you don't need it to run.

## Layout

```
benchmark/
├── _scenarios/scenario-NN-<slug>/   SOURCE OF TRUTH — 25 scenario templates
│   ├── scenario.toml                query, params, prompt, 4-class adjudication criteria
│   ├── rubrics.toml                 process rubric (signed-weight MUST-DO / MUST-NOT items)
│   └── policy_rubrics.toml          policy-relevance rubric
├── _template/                       the Harbor task template + the in-container verifier
├── corpus/                          policy corpus (assembled locally, not in git — see corpus/README.md)
├── corpus_public/                   the 16 corpus documents that are US government works
├── corpus_manifest.csv              where each of the 67 corpus documents comes from
├── eval/                            offline scoring: process rubric, grounding, judge panel
└── scripts/                         build_tasks.py and friends
```

Tasks are **generated**, never hand-authored. `_scenarios/` is the source of truth; `tasks/` is a
build artifact and is gitignored — don't edit or commit it.

### What a scenario holds

- `scenario.toml` — `id`, `description`, `params` (the ID columns the prompt needs), `prompt` (the
  question, with `{param}` placeholders), and the spec fields used for labelling and grading:
  `clinical_question`, the four 4-class label rules (`inclusion_criteria` → Yes,
  `exclusion_criteria` → No, `indeterminate_lack_of_data_criteria`,
  `indeterminate_medical_ambiguity_criteria`), plus `required_evidence` and
  `grounding_requirements`. **Only `prompt` is ever shown to the agent.**
- `rubrics.toml` — the process rubric: `[[rubric]]` entries with `id`, `title`, and a signed
  `weight` (`+N` = MUST-DO, credited when present; `-N` = MUST-NOT, penalised when present). It
  grades methodology from the trajectory, independent of the outcome label. A scenario without one
  still builds; the process axis is skipped for it.
- `policy_rubrics.toml` — the scenario's policy-relevance rubric.

**The patient cohort is not in the scenario dirs.** Which patients each scenario covers, and their
gold labels, live in the gated bundle (`full750.csv`) — see [DATA_ACCESS.md](../DATA_ACCESS.md).
The SQL that defined each cohort is in [`../construction/cohorts/`](../construction/cohorts/), and
it is a sampling stratifier, never ground truth.

## Build and run

```bash
# once, and again after any change to the Dockerfile or packages/mimic_tools/
bash ../scripts/build_image.sh

set -a; source ../.env; set +a          # plain KEY=VAL — must be exported, not just sourced

python scripts/build_tasks.py           # 750 tasks -> tasks/
python scripts/build_dev148.py          # or the 148-case development subset

python ../scripts/run_sweep.py --systems leaderboard --tasks benchmark/tasks
python eval/report/score_clinicare.py jobs/<job> --csv out.csv
```

`build_tasks.py` reads the cohort from the gated bundle (`CLINICARE_COHORT_DIR`) and injects
three things into every task at render time, none of which are committed: the trial image ref
(a content hash — see `scripts/env_image.py`), the egress allowlist, and the scenario's rubric
bundle. See [`../DATA_ACCESS.md`](../DATA_ACCESS.md) for the bundle and the MIMIC data.

Useful flags: `--scenarios <id ...>` for a partial build, `--no-policy` for the policy-off
ablation, `--include-answer` to bake the rubric into the prompt (that's the oracle build, used
during construction, not for evaluating a candidate).

On the policy-on build the agent gets three read-only tools over the guideline corpus —
`list_policy_docs` (glob), `search_policy` (grep, returning 1-indexed line numbers) and
`read_policy` (a line range) — and cites passages inline by document and line span. Retrieval is
deliberately agentic grep/read with no embeddings and no vector store, so every benchmarked agent
sees the same citation interface and the trace keeps line-level provenance. `--no-policy` moves
three things together: those tools leave the MCP registry (`MIMIC_POLICY_TOOLS=0`), the corpus is
not staged into the container, and the citation section is dropped from the instruction — so the
ablation measures whether corpus access changes the verdict, as against the model answering from
parametric recall.

### The 16 systems

`--systems leaderboard` runs the paper's main table: **10 production CLIs on their own vendor
models** (Claude Code ×2, Codex ×5, Gemini CLI ×3) plus **6 open-weight models on `opencode`**,
the neutral harness. `--systems ablation` adds the three systems that appear only in the ablation
sections (Opus 4.8 for policy-on/off, repeated-run reliability and reasoning effort; qwen-coder
and mini-swe-agent for the fixed-model harness comparison).

**The six open-weight systems need a gateway.** Their model ids are `openai/fireworks_ai/<model>`,
which no vendor endpoint serves — set `OPENAI_BASE_URL` to your own OpenAI-compatible proxy and
build the tasks with it exported: `build_tasks.py` allowlists that host for you. Because the
allowlist is baked in at build time, tasks built before you set it cannot reach the gateway.
The ten native systems need only the vendor API keys.

`--systems medhelm` runs five models MedHELM evaluates that postdate the paper, set up by the same
rule: Claude Opus 5.5 on Claude Code, GPT-6 Astra on Codex, Gemini 3.8 Flash on Gemini CLI, and
Qwen3.8 Max and Muse Spark 1.3, which have no CLI of their own, on `opencode` through the gateway.

### Keeping runs apart

Each build lands in its own directory, so no build overwrites another:

| build | directory |
| :-- | :-- |
| full, policy-on (the default) | `tasks/` |
| full, policy-off | `tasks_nopolicy/` |
| dev subset (`build_dev148.py`) | `tasks_dev_policy/` or `tasks_dev_nopolicy/` |
| answer mode (`--include-answer`, label generation) | `tasks_answer/`, `tasks_answer_nopolicy/` |

Job names, however, are derived from the dir name and both `tasks/` and `tasks_dev_policy/`
reduce to the same bare `cbench_sweep_<system>`, so a dev sweep collides with a full-750 sweep of the
same system unless you pass `--job-suffix`. `run_sweep.py --tasks` defaults to the full build, so
omitting it on a dev run sweeps the wrong cohort silently.

## How scoring splits

Two halves, deliberately:

**In-trial, deterministic.** `_template/tests/test.sh` runs `judge.py` with **no LLM, no
credentials, and no network**. It matches the agent's `## Answer:` verdict against the gold
label, then computes cited-ID grounding and tool-call health from the trial's own trajectory.
That's what makes the primary score reproducible and hard to game — and why the agent container
never needs judge credentials, which it could otherwise read out of its own environment.

**Offline, LLM-judged.** The process rubric and policy citation support are scored afterwards by
`eval/`, from artifacts the trial persisted. Same inputs, so a re-grade needs no re-run.

Policy grounding is a third axis, reported separately and never blended into the primary `score`.
The paper defines what it measures; `eval/` is where it is computed.

## Scenarios and tasks

| Unit | In code | In the paper | What it is |
| --- | --- | --- | --- |
| Template | `scenario` | scenario | One `_scenarios/` dir. **25** of them. |
| Instantiation | `task` | case | One scenario × one patient → one Harbor task, id `<scenario_id>-<param-hash>`. 30 per scenario, **750** total. |

Task ids hash the cohort row's parameter values **in the order `scenario.toml` declares them**.
Two scenarios declare the same three parameters in different orders, so that ordering is
load-bearing — changing it renumbers tasks and silently invalidates every result keyed by id.
The hashing lives in `clinicare_core.taskids`, in one copy, for exactly that reason.
