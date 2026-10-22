# Contributing

[README.md](README.md) covers what the benchmark is and how to run it;
[DATA_ACCESS.md](DATA_ACCESS.md) covers credentialing and preparing MIMIC-IV. This file is the
short list of things you can break that the code will not stop you from breaking.

## Do not commit

`.gitignore` blocks all of these — it names the sensitive files individually rather than relying
on a glob, on purpose. Please keep it that way when you add to it.

- **Patient data and gold labels** — `full750.csv`, `dev148.csv`, `patients.csv`,
  `silver_patients.csv`, any `.csv.gz`. MIMIC derivatives under the PhysioNet DUA.
- **Trial outputs** — `jobs/` quotes the patient record, and task ids are an unsalted hash of a
  case's parameters, so even a bare verdict table re-identifies patients to anyone holding this
  repo. Don't attach one to an issue or a PR either.
- **Built tasks** — `benchmark/tasks*/` is generated. Edit `benchmark/_scenarios/` instead.
- **`.env`** — only `.env.template` is tracked, with placeholders only.

If you think you may already have committed one of these, say so before pushing. A rewrite while
the history is local is cheap; after publication it is not.

## Adding a corpus document means adding a manifest row, not a file

`benchmark/corpus_public/` holds exactly the documents that are US government works — public
domain, redistributable. Every other document in the corpus is fetch-only: we ship the recipe and
each user assembles their own copy under the terms governing that document.

So a new document is a row in `benchmark/corpus_manifest.csv` with a `source_sha256` and an
`action` of `fetch-pdf` / `fetch-html` / `download-pdf` / `download-epub`. Only `ship-md` puts
bytes in the repo, and only for a public-domain government work. Record provenance in
[THIRD_PARTY.md](THIRD_PARTY.md); see [benchmark/corpus/README.md](benchmark/corpus/README.md)
for how acquisition works.

## Two invariants the tests won't catch

**Data access stays credential-isolated.** Don't add an in-container path to the data, however
convenient. A code-capable agent that can reach the credentials can query the raw tables and
bypass the governed tool surface, and a submission that can do that is not comparable to one that
cannot. `run_sql` is confined at the DuckDB connection rather than by inspecting SQL text for the
same reason — a text check is not sufficient there.
See [`packages/mimic_tools/SIDECAR.md`](packages/mimic_tools/SIDECAR.md).

**Tool descriptions are descriptive, not prescriptive.** A tool's docstring states its scope and
its limits. It must not name a workaround that routes through another tool, and it must not
mention an entity from any specific case. The first tells the agent how to investigate, which is
part of what is being measured; the second overfits the surface to the cases that exist today.

## Before you open a PR

```bash
uvx ruff format . && uvx ruff check .

# Scoring and grounding. Runs offline against fixtures — no credentials, no network, no spend.
# (Set JUDGE_BASE_URL/JUDGE_API_KEY/JUDGE_MODEL to additionally exercise the LLM-judged paths.)
for t in benchmark/eval/tests/test_*.py scripts/tests/test_*.py; do uv run python "$t" || break; done

# The tool surface, against your Parquet tree. Exercises every tool on one patient.
MIMIC_DATA_ROOT=/path/to/mimic-parquet uv run python packages/mimic_tools/smoke_test.py

# Only if you touched the Dockerfile or packages/mimic_tools/ — the image tag is a content hash
# of exactly those inputs, so the built tasks and the image cannot silently disagree.
bash scripts/build_image.sh
```

`test_verdict.py` pins the contract behind the primary reward, and every clause in it exists
because a naive parser would let an agent score without committing to an answer. If a change to
`judge.py` makes it fail, the test is probably right.

## Reporting a security or privacy issue

Please don't open a public issue. Use this repository's private vulnerability reporting
(Security → Report a vulnerability) so a fix can land before the details are public. That applies
equally to anything that looks like exposed patient data or a way around the credential isolation
above.
