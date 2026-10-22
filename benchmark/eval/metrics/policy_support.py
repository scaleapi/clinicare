"""Post-hoc policy citation-support scorer — the LLM half of policy grounding.

The policy outcome rubric has three items. Relevance (PR1, gold-doc set membership) and citation
existence (PC1, does each citation resolve to a real corpus document and a valid line span) are
deterministic and graded in-trial. The third — citation SUPPORT (PC2), "does this cited passage
factually back the claim it is attached to?" — needs an LLM, so it is graded here instead, over
``verifier/report.md``. The rubric is then re-scored with the support results folded in, so a
trial's policy outcome pct is not final until this has run.

Cost scales with CITATIONS, not trials: ``policy_scoring.judge_support`` runs once per resolving
citation, so a verbose citer costs many times what a sparse one does.

    export JUDGE_BASE_URL=... JUDGE_API_KEY=...
    python benchmark/eval/metrics/policy_support.py jobs/<job> --dry-run   # coverage, no LLM
    python benchmark/eval/metrics/policy_support.py jobs/<job>
    python benchmark/eval/metrics/policy_support.py jobs/<job> --diff      # vs in-trial

Writes ``<trial>/verifier/policy_support.json`` (idempotent; ``--force`` to re-grade) and
``<job>/policy_support_summary.json``. Same coverage-gate contract as process.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clinicare_core import write_reward_atomic
from clinicare_core.paths import bootstrap

EVAL = Path(__file__).resolve().parents[1]  # benchmark/eval

bootstrap(__file__, "metrics", "container_tests")
import process as SP  # noqa: E402  (shared trial classifier + coverage contract)

HEALTHCARE = EVAL.parent
import policy_scoring as PS  # noqa: E402

# The corpus is assembled locally, not repo content (benchmark/corpus/README.md);
# CLINICARE_CORPUS_DIR points at the directory acquire_corpus.py built.
DEFAULT_CORPUS = Path(os.environ.get("CLINICARE_CORPUS_DIR") or (HEALTHCARE / "corpus"))

SCORER_VERSION = 1


def _expect(rec: dict, judge_model: str, corpus: Path) -> dict:
    """Identity of a policy_support grading: scorer version, judge model (effort included), the
    corpus it resolved citations against, and a hash of the report + policy rubrics.

    Note the corpus enters as its PATH, not a hash of its 67 documents: pointing
    CLINICARE_CORPUS_DIR somewhere else invalidates, but editing a document in place does not.
    Hashing the corpus per trial would be wasteful and it is shared across all of them — if you
    revise corpus text, use --force.
    """
    return {
        "scorer_version": SCORER_VERSION,
        "judge_model": judge_model,
        "judge_base_url": os.environ.get("JUDGE_BASE_URL", ""),
        "corpus": str(corpus),
        "inputs_sha256": SP.inputs_sha(rec, "policy_rubrics"),
    }


def grade_one(rec: dict, corpus: Path, creds: dict, batch_size: int = 0) -> dict:
    """Judge every resolving citation in one report, then re-grade the outcome rubric with them."""
    bundle = rec["bundle"]
    report = SP.JP.J._read_text(str(rec["report"]))
    gold = {PS._norm_doc(d) for d in (bundle.get("policy_gold_docs") or [])}
    gold.discard(None)

    citations = PS.parse_citations(report)
    resolved = [PS.resolve_citation(c, corpus) for c in citations]

    out: dict = {
        "schema": "healthcare/policy_support.v1",
        "scorer_version": SCORER_VERSION,
        "judge_model": creds["model"]
        + (f"@{creds['reasoning_effort']}" if creds.get("reasoning_effort") else ""),
        "task_id": rec["task_id"],
        # `case_id`: pre-rename spelling, still present in bundles captured under jobs/.
        "scenario_id": bundle.get("scenario_id") or bundle.get("case_id"),
        "corpus": str(corpus),
        # Exact input identity, checked by SP.cached_ok (see _expect).
        "inputs_sha256": SP.inputs_sha(rec, "policy_rubrics"),
        "judge_base_url": creds.get("base_url", ""),
        "policy_citations_total": len(citations),
        "policy_citations_in_corpus": sum(
            1 for r in resolved if r["in_corpus"] and r["span_valid"]
        ),
    }

    # Unresolvable citations get None (there is no passage to judge), which is what
    # grade_policy_outcome expects. Resolving ones are judged either one-per-call (batch_size=0,
    # the reference path) or batched — see judge_support_batch on why the two can differ.
    out["support_mode"] = f"batch{batch_size}" if batch_size else "per-citation"
    judgeable = [i for i, r in enumerate(resolved) if r["in_corpus"] and r["span_valid"]]
    support: list[dict | None] = [None] * len(citations)
    try:
        if batch_size:
            verdicts = PS.judge_support_batch(
                [(citations[i].sentence, resolved[i]["cited_text"]) for i in judgeable],
                creds,
                batch_size=batch_size,
            )
            for i, v in zip(judgeable, verdicts):
                support[i] = v
            out["n_judge_calls"] = -(-len(judgeable) // batch_size)
        else:
            for i in judgeable:
                support[i] = PS.judge_support(
                    citations[i].sentence, resolved[i]["cited_text"], creds
                )
            out["n_judge_calls"] = len(judgeable)
    except BaseException as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        return out

    judged = [s for s in support if s is not None]
    n_sup = sum(1 for s in judged if s.get("supported"))
    out["n_judged"] = len(judged)
    out["n_supported"] = n_sup
    out["policy_citation_support_rate"] = round(n_sup / len(judged), 3) if judged else None
    out["policy_citation_correct_rate"] = round(n_sup / len(citations), 3) if citations else None

    outcome = PS.grade_policy_outcome(
        True,  # policy-ON only; the OFF path is deterministic and stays in-trial
        citations,
        resolved,
        gold,
        report,
        bundle.get("policy_rubrics") or [],
        bundle.get("prompt", ""),
        creds,
        support_results=support,
    )
    if outcome is None:
        out["error"] = "grade_policy_outcome returned None (no authored policy_rubrics?)"
        return out
    out["outcome_graded"] = outcome["graded"]
    out["score"] = outcome["score"]
    out["citations"] = [
        {
            "raw": c.raw,
            "doc": c.doc,
            "lines": [c.line_start, c.line_end],
            "in_corpus": r["in_corpus"],
            "span_valid": r["span_valid"],
            "sentence": c.sentence[:300],
            "support": s,
        }
        for c, r, s in zip(citations, resolved, support)
    ]
    return out


def amend_reward(trial: Path, payload: dict) -> bool:
    """Merge the post-hoc policy numbers into reward.json (opt-in; mutates a Harbor artifact).

    Also overwrites ``policy_outcome_pct`` with the COMPLETE (support-included) value and sets
    ``policy_support_judged``, so a consumer can no longer average a deterministic-only 2-item mean
    together with a complete 3-item one under the same key. Every scenario ships 3 equal-weight policy
    items, so that difference is worth up to 33.3 pts.
    """
    rp = trial / "verifier" / "reward.json"
    try:
        reward = json.loads(rp.read_text())
    except (OSError, ValueError):
        return False
    sc = payload.get("score") or {}
    reward["policy_support_judged"] = 1
    reward["policy_outcome_items_scored"] = sum(
        1 for g in (payload.get("outcome_graded") or []) if g.get("score") is not None
    )
    if sc.get("pct") is not None:
        reward["policy_outcome_pct"] = sc["pct"]
        reward["policy_outcome_earned"] = sc.get("earned", 0.0)
        # The outcome is now COMPLETE — drop the deterministic-only value so the two can never
        # both be present (and so no consumer sees a stale partial number beside the real one).
        reward.pop("policy_outcome_pct_deterministic", None)
    for key in ("policy_citation_support_rate", "policy_citation_correct_rate"):
        if payload.get(key) is not None:
            reward[key] = payload[key]
    write_reward_atomic(rp, reward)  # a run_case.py sharing this job may be reading it
    return True


def diff_against_in_trial(trial: Path, payload: dict) -> dict | None:
    """Compare the post-hoc support rate + outcome pct against what the in-trial judge wrote."""
    try:
        reward = json.loads((trial / "verifier" / "reward.json").read_text())
    except (OSError, ValueError):
        return None
    old_rate = reward.get("policy_citation_support_rate")
    old_pct = reward.get("policy_outcome_pct")
    if old_rate is None and old_pct is None:
        return None
    return {
        "in_trial_support_rate": old_rate,
        "post_hoc_support_rate": payload.get("policy_citation_support_rate"),
        "in_trial_outcome_pct": old_pct,
        "post_hoc_outcome_pct": (payload.get("score") or {}).get("pct"),
        "n_citations": payload.get("policy_citations_total"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("jobs", nargs="+", help="job dirs or globs")
    ap.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help=f"policy corpus dir (default {DEFAULT_CORPUS})",
    )
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", "openai/gpt-5.5"),
        help="'model' or 'model@effort'",
    )
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="judge N citations per LLM call instead of one each (0 = per-citation, the reference "
        "path). Batching cuts calls ~N-fold but can shift verdicts — validate agreement against "
        "the default before using batched numbers in a report.",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="classify + report coverage, no LLM")
    ap.add_argument(
        "--amend-reward",
        action="store_true",
        help="merge the support rate + completed outcome back into each trial's reward.json "
        "(mutates it). Without this, metric.py's mean_policy_citation_support_rate and the policy "
        "summary's support column stay blank even after this scorer runs.",
    )
    ap.add_argument(
        "--diff", action="store_true", help="compare against the in-trial policy numbers"
    )
    ap.add_argument("--rubrics-from", default=None, metavar="TASKS_DIR")
    ap.add_argument("--limit", type=int, default=0, help="cap trials per job (suppresses the gate)")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not (corpus.is_dir() and any(corpus.glob("*.md"))):
        print(f"ERROR: corpus dir empty or missing: {corpus}", file=sys.stderr)
        return 1

    job_dirs = sorted({d for pat in args.jobs for d in glob.glob(pat) if os.path.isdir(d)})
    if not job_dirs:
        print("no job dirs matched", file=sys.stderr)
        return 1

    model, _, effort = args.judge_model.partition("@")
    creds = {
        "model": model,
        "reasoning_effort": effort or None,
        "base_url": os.environ.get("JUDGE_BASE_URL", ""),
        "api_key": os.environ.get("JUDGE_API_KEY", ""),
    }
    if not args.dry_run and not (creds["base_url"] and creds["api_key"]):
        print(
            "ERROR: JUDGE_BASE_URL and JUDGE_API_KEY must be set (or use --dry-run)",
            file=sys.stderr,
        )
        return 1

    rubrics_from = Path(args.rubrics_from) if args.rubrics_from else None
    gate_failures = 0
    empty_jobs = []

    for job in job_dirs:
        jd = Path(job)
        trials = [
            SP.classify(t, rubrics_from, require="policy_rubrics")
            for t in sorted(jd.glob("*__*"))
            if t.is_dir()
        ]
        # No `<task>__<hash>` children = wrong path (a TRIAL dir rather than the JOB dir above it,
        # or a typo). Fail rather than printing a tidy "scored 0/0" and exiting 0, which reads as
        # a real result. Mirrors the same guard in process.py.
        if not trials:
            empty_jobs.append(jd)
            continue
        # Policy-OFF trials are correctly scored in-trial (deterministic abstention check) — not
        # a coverage failure, and nothing here to judge.
        for rec in trials:
            if not rec["reason"] and not rec["bundle"].get("policy_expected"):
                rec["reason"] = "policy_off"

        n_trials = len(trials)
        if args.limit:
            trials = trials[: args.limit]

        todo, cached, skipped = [], [], []
        for rec in trials:
            if rec["reason"]:
                skipped.append(rec)
            elif (
                SP.cached_ok(
                    rec["path"] / "verifier" / "policy_support.json",
                    _expect(rec, args.judge_model, corpus),
                )
                and not args.force
            ):
                cached.append(rec)
            else:
                todo.append(rec)

        print(
            f"\n=== {jd.name}: {n_trials} trial dirs — {len(todo)} to grade, {len(cached)} cached, {len(skipped)} skipped"
        )
        if args.limit:
            print(f"    ⚠️  PARTIAL RUN: --limit {args.limit} — coverage gate suppressed")

        results: dict[str, dict] = {}
        if args.dry_run:
            print("    --dry-run: no grading performed")
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = {
                    ex.submit(grade_one, rec, corpus, creds, args.batch_size): rec for rec in todo
                }
                for i, f in enumerate(as_completed(futs), 1):
                    rec = futs[f]
                    payload = f.result()
                    (rec["path"] / "verifier").mkdir(parents=True, exist_ok=True)
                    (rec["path"] / "verifier" / "policy_support.json").write_text(
                        json.dumps(payload, indent=2)
                    )
                    results[rec["trial"]] = payload
                    if args.amend_reward and not payload.get("error"):
                        amend_reward(rec["path"], payload)
                    if i % 25 == 0:
                        print(f"    {i}/{len(todo)} graded", file=sys.stderr)
        for rec in cached:
            try:
                results[rec["trial"]] = json.loads(
                    (rec["path"] / "verifier" / "policy_support.json").read_text()
                )
            except (OSError, ValueError):
                results[rec["trial"]] = {"error": "unreadable cached policy_support.json"}
        # Amend CACHED trials too: policy_support.json is written on the first (un-amended) run, so a
        # later --amend-reward would otherwise find everything cached and silently amend nothing.
        if args.amend_reward:
            n_amended = sum(
                1
                for rec in cached
                if not (results.get(rec["trial"]) or {}).get("error")
                and amend_reward(rec["path"], results.get(rec["trial"]) or {})
            )
            if n_amended:
                print(f"    amended {n_amended} cached trial reward.json")

        errored = [t for t, p in results.items() if p.get("error")]
        rates = [
            p["policy_citation_support_rate"]
            for p in results.values()
            if p.get("policy_citation_support_rate") is not None
        ]
        pcts = [p["score"]["pct"] for p in results.values() if p.get("score")]
        n_cited = sum(p.get("policy_citations_total") or 0 for p in results.values())
        by_reason: dict[str, int] = {}
        for rec in skipped:
            by_reason[rec["reason"]] = by_reason.get(rec["reason"], 0) + 1

        print(
            f"    scored {len(pcts)}/{n_trials if not args.limit else len(trials)}  citations judged={n_cited}"
        )
        if rates:
            print(
                f"    mean support rate = {sum(rates) / len(rates):.3f}   mean policy_outcome_pct = {sum(pcts) / len(pcts):.1f}"
            )
        if by_reason:
            print(f"    skipped: {by_reason}")
        if errored:
            print(f"    ⚠️  {len(errored)} error(s): {errored[:3]}")

        diffs = []
        if args.diff:
            for trial_name, payload in results.items():
                d = diff_against_in_trial(jd / trial_name, payload)
                if d:
                    diffs.append((trial_name, d))
            if diffs:
                same = sum(
                    1 for _, d in diffs if d["in_trial_outcome_pct"] == d["post_hoc_outcome_pct"]
                )
                print(
                    f"    parity vs in-trial: {len(diffs)} comparable — {same} identical outcome_pct"
                )
                for t, d in diffs[:5]:
                    print(
                        f"      {t}: support {d['in_trial_support_rate']} → {d['post_hoc_support_rate']}"
                        f"  outcome {d['in_trial_outcome_pct']} → {d['post_hoc_outcome_pct']}"
                        f"  ({d['n_citations']} citations)"
                    )
            else:
                print("    parity: no in-trial policy numbers to compare against")

        summary = {
            "schema": "healthcare/policy_support_summary.v1",
            "scorer_version": SCORER_VERSION,
            "job": jd.name,
            "judge_model": args.judge_model,
            "corpus": str(corpus),
            "n_trial_dirs": n_trials,
            "n_scored": len(pcts),
            "n_cached": len(cached),
            "n_judge_errors": len(errored),
            "n_citations_judged": n_cited,
            "support_mode": f"batch{args.batch_size}" if args.batch_size else "per-citation",
            "n_judge_calls": sum(p.get("n_judge_calls") or 0 for p in results.values()),
            "skipped": by_reason,
            "partial_limit": args.limit or None,
            "mean_support_rate": (sum(rates) / len(rates)) if rates else None,
            "mean_policy_outcome_pct": (sum(pcts) / len(pcts)) if pcts else None,
        }
        if not args.dry_run:
            (jd / "policy_support_summary.json").write_text(json.dumps(summary, indent=2))

        # Derive from the shared SP.FAILURE_REASONS rather than listing reasons here. A gate that
        # enumerates its own reasons drifts, and a reason left out means a job whose verifier
        # never ran exits 0 with an incomplete summary.
        bad = len(errored) + sum(by_reason.get(r, 0) for r in SP.FAILURE_REASONS)
        if bad and not args.limit:
            gate_failures += bad

    if empty_jobs:
        print(
            "\n❌ no trial dirs (`<task>__<hash>/`) found in: "
            + ", ".join(str(d) for d in empty_jobs)
            + "\n   Pass the JOB dir, not a trial dir — e.g. jobs/<job>, not jobs/<job>/<task>__<hash>.",
            file=sys.stderr,
        )
        return 2
    if gate_failures:
        print(
            f"\n❌ coverage gate: {gate_failures} trial(s) unscored for a fixable reason.",
            file=sys.stderr,
        )
        return 0 if args.allow_incomplete else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
