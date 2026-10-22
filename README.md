# CliniCARE-Bench

A clinical agent benchmark over **MIMIC-IV**. An agent is given a clinical question about one
real patient record, reaches the chart through a governed tool surface, and must return a
four-way verdict with cited evidence:

```
YES · NO · INDETERMINATE — lack of data · INDETERMINATE — medical ambiguity
```

Getting the verdict right is one axis of several. A determination is also scored on the
investigation behind it: whether the agent retrieved the evidence the case requires, cited passages
that support the claims they are attached to, followed the scenario's process rubric, and deferred
when the record genuinely cannot settle the question. That last axis is what the two indeterminate
classes carry — 150 of the 750 cases admit no defensible Yes or No (123 lack-of-data, 27 medical
ambiguity), so confidence alone does not score well here.

## Units

| Unit | In code | In the paper | What it is |
| --- | --- | --- | --- |
| Template | `scenario` | scenario | One `benchmark/_scenarios/scenario-NN-<slug>/` — clinical query, four-way adjudication criteria, evidence requirements. **25** of them. |
| Instantiation | `task` | case | One scenario × one MIMIC-IV patient → one Harbor task, id `<scenario_id>-<param-hash>`. 30 per scenario, **750** total. |

`case` and `task` name the same thing; prefer `task` in code and read `case` in prose.

## What you need

CliniCARE runs against real patient records, so **you must be credentialed for MIMIC-IV on
PhysioNet**. Nothing in this repository is a substitute for that, and no patient data is
distributed here.

There are three artifacts, in two tiers:

| Artifact | Contains | Rights | Who can get it |
| --- | --- | --- | --- |
| **This repository** | Scenarios, tool surface, eval pipeline | MIT (see [LICENSE](LICENSE)) | Public |
| **Cohort bundle** | `full750.csv`, `dev148.csv` — which patients, and their gold labels | MIMIC derivative, PhysioNet DUA | Credentialed |
| **Policy corpus** | 67 clinical guideline documents for the policy-grounding axis | Third-party, per-publisher terms | Anyone — assembled locally |

The cohort bundle comes from MedHELM, which verifies PhysioNet credentialing. The corpus does
not: publisher terms let you fetch a copy but not redistribute one, so the repository ships the
acquisition recipe — a manifest of sources plus the 16 documents that are US government works —
and `construction/scripts/acquire_corpus.py` assembles the rest on your machine, under the terms
governing each document. About 18 of them have to be downloaded through a browser, because their
publishers block scripted access. See
[DATA_ACCESS.md](DATA_ACCESS.md#5-build-the-policy-corpus).

The MIT grant covers everything this project authored, data as well as code — that includes the
scenario specifications under `benchmark/_scenarios/` (`scenario.toml`, `rubrics.toml`,
`policy_rubrics.toml`), the task templates and the judge prompts, not only the Python. The one
exception is third-party content, which MIT cannot cover: the 16 US Government guideline documents
shipped under `benchmark/corpus_public/` are public-domain works redistributed as a convenience.
[THIRD_PARTY.md](THIRD_PARTY.md) records the provenance of every corpus document.

Reproducing a published number means re-running the agents against MIMIC and scoring the result,
so it needs credentials. Per-trial outputs are not redistributed: a trial's report and trajectory
quote the patient record, and even a stripped-down verdict table would carry task ids, which are
an unsalted hash of the case's parameters and so identify the patient for anyone holding this
repo.

See **[DATA_ACCESS.md](DATA_ACCESS.md)** for credentialing, obtaining MIMIC-IV, and preparing it.

## Layout

```
clinicare/
├── benchmark/           # run the benchmark
│   ├── _scenarios/      # SOURCE OF TRUTH — 25 scenario templates
│   ├── _template/       # Harbor task template + the in-container verifier
│   ├── corpus/          # EMPTY until you unpack the corpus bundle here (README only)
│   ├── eval/            # offline scoring: process rubric, grounding, judge reliability
│   └── scripts/         # build_tasks.py and friends
├── construction/        # how the benchmark was BUILT — not needed to run it
├── packages/
│   ├── mimic_tools/     # the MIMIC tool surface the agent calls
│   └── clinicare_core/  # shared helpers + the per-CLI harness shims (agents/)
└── scripts/             # sweep driver + scoring/reporting
```

Tasks are **generated**, not hand-authored. `benchmark/_scenarios/` is the source of truth;
`tasks/` is a build artifact and is gitignored — never edit or commit it.

## Quickstart

Needs **Python 3.12+** (harbor's floor) and Docker. `uv sync` on 3.11 fails with an
unsatisfiable-resolution error rather than a version message.

```bash
uv sync                                    # add --extra modal to run trials on Modal
cp .env.template .env                      # then fill it in
set -a; source .env; set +a                # must be exported, not just sourced

bash scripts/build_image.sh                           # the trial image, once

uv run python benchmark/scripts/build_tasks.py        # 750 tasks
uv run python benchmark/scripts/build_dev148.py       # or the 148-case dev subset

uv run python scripts/run_sweep.py --systems claude-opus5      # one system; --systems leaderboard for all 16
uv run python benchmark/eval/grade.py jobs/<your-job-dir>   # every offline scorer + the table
```

Run trials through `run_sweep.py` rather than calling `harbor run` directly: it merges the
sidecar compose overlay that starts the credentialed mimic-tools service, without which the agent
has no MIMIC access at all. `--dry-run` prints the exact harbor command it would issue.

`grade.py` runs the LLM-judged and grounding metrics and then the report; add `--skip-llm` for
the deterministic subset only (no judge credentials, no spend).

### Run one case (for wrappers such as MedHELM)

`scripts/run_case.py` runs a single task with one system into a job dir you name, and prints the
trial as one JSON line. It builds the same harbor command as `run_sweep.py` and loads `.env`
itself, so a wrapper needs only a system tag and a task id:

```bash
uv run python scripts/run_case.py --system codex-gpt56sol --task-id <task id> \
    --jobs-dir jobs/medhelm --job-name <unique name>      # add --grade for the LLM-judged axes
```

```json
{"status": "ok", "trial_dir": "...", "task_id": "...", "model_name": "openai/gpt-5.6-sol",
 "agent_name": "...", "agent_version": "...", "exception_type": null, "task_checksum": "...",
 "graded": true,
 "grading": {"process": {"outcome": "scored", "reason": "scored", "detail": null},
             "policy": {"outcome": "skipped", "reason": "policy_off", "detail": null}}}
```

`status` is `ok` (scored), `agent_error` (scored, but the agent raised, e.g. a timeout), or
`dead` (setup or environment failure, so unscored). Harbor exits 0 in all three cases, so read
`status`, not the exit code. With `--grade`, `grading` says what each LLM judge did and why, so a failure can be traced to its cause: `skipped` is by design (`no_report` means the agent wrote no report; `policy_off` and `no_*_rubrics` mean there was nothing to score), and `failed` names a judge credentials problem (`judge_credentials_missing`, `judge_credentials_rejected`), a judge or API error (`judge_error`, with the exception in `detail`), a task-build problem (`no_rubrics`), or a crashed judge script (`judge_script_failed`). `graded` is true when no judge failed. The full list is in the `run_case.py` docstring. Exit 2 means there is no trustworthy trial to report: a usage or
preflight error, harbor refusing the job, a trial whose `result.json` or `reward.json` cannot be
parsed, or a finished trial built from a different version of the task. Re-running with the same `--job-name` returns the finished trial without re-running it,
but only while the task is unchanged (harbor's task checksum). After a CliniCARE update or a task
rebuild, use a new `--job-name`.

## How scoring works

The **outcome** score is deterministic and needs no LLM and no credentials: the in-container
verifier matches the agent's `## Answer:` verdict against the gold label, case-insensitively.
Cited-ID grounding and tool-call health are computed the same way, from the trial's own
trajectory.

The **LLM-judged** metrics — the process rubric and policy citation support — are scored
afterwards from the persisted artifacts by `benchmark/eval/`, not inside the trial. Keeping them
out of the trial is what lets the outcome score stay reproducible and un-gameable. Grounding and
policy are reported as their own axes and never blended into the primary score.

## Data access is credential-isolated

The agent container holds **no** data credentials and has no route to the data store. Every
tool call is proxied to a sidecar that does. This is not incidental hardening: without it, a
code-capable agent can read the credentials out of its own environment and query the raw tables
directly, bypassing the governed tool surface.
See [`packages/mimic_tools/SIDECAR.md`](packages/mimic_tools/SIDECAR.md).

## Status

Being prepared for public release. The code is MIT ([LICENSE](LICENSE)); the cohort and corpus
bundles are distributed separately to PhysioNet-credentialed users — see
[DATA_ACCESS.md](DATA_ACCESS.md).
