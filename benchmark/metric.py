#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Harbor job-level metric for the MIMIC healthcare multi-key rewards.

Why this exists:
    Harbor's built-in `mean`/`sum`/… metrics each require the per-trial reward dict to
    contain exactly one key (see `harbor.metrics.mean.Mean`). The healthcare verifier
    (`_template/tests/judge.py`) emits many keys — the blended `score`, the deterministic
    `outcome_correct`, the LLM `process_pct`, and the policy-grounding block
    (`policy_outcome_pct`, `policy_file_recall`, `policy_citation_*`, …). So the default
    aggregator raises, and we aggregate here instead.

Wire it into a job YAML:

    metrics:
      - type: uv-script
        kwargs:
          script_path: benchmark/metric.py

Interface (matches `harbor.metrics.uv_script.UvScript`):
    uv run metric.py -i rewards.jsonl -o metric.json
    `rewards.jsonl` = one trial's reward dict per line (or literal `null` if the trial
    failed before verification); `metric.json` = a flat {name: float|int} dict.

A reward whose verifier failed before scoring (``rubrics_load_failed`` / ``judge_launch_failed``)
is treated like a ``null``: it counts in ``n_trials`` and ``n_verifier_failed``, not in
``n_completed`` or any mean, because its ``score: 0.0`` is a placeholder rather than a verdict.

Rates that a trial leaves undefined (omitted from its reward dict) are averaged only over
the trials that reported them. Policy outcome is split by condition (policy-on vs -off)
since the expected behavior differs.

A key NO trial reported is OMITTED from metric.json entirely rather than emitted as 0.0. That
matters because the in-trial verifier is deterministic-only (the top-level README,
"Current placement"):
``process_pct`` and ``policy_citation_support_rate`` are now produced post-hoc by
eval/metrics/{process,policy_support}.py, so a fresh job has no such key. Averaging an empty
column to 0.0 would report "every system scored zero on process" instead of "not measured here" —
the exact silent-zero this file is meant to avoid. The keys are still read, so a corpus whose
rewards were amended post-hoc (``--amend-reward``) still aggregates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# clinicare_core.rewards.VERIFIER_FAILED_KEYS, kept local because harbor runs this file as a
# dependency-free uv script. A verifier that fails before scoring still writes score 0.0, so a
# reward carrying one of these keys is unscored, not a wrong answer. A test checks the copies match.
VERIFIER_FAILED_KEYS = ("rubrics_load_failed", "judge_launch_failed")


def _mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    args = p.parse_args()

    rewards: list[dict] = []
    n_trials = 0
    n_unscored = 0
    for raw in args.input.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        n_trials += 1
        if line == "null":
            continue
        reward = json.loads(line)
        if any(reward.get(key) == 1 for key in VERIFIER_FAILED_KEYS):
            n_unscored += 1  # counted in n_trials, kept out of n_completed and every mean
            continue
        rewards.append(reward)

    def col(key: str) -> list[float]:
        """Values of `key` across trials that reported it (skips undefined/omitted)."""
        return [float(r[key]) for r in rewards if key in r and r[key] is not None]

    on = [r for r in rewards if int(r.get("policy_expected", 0)) == 1]
    off = [
        r for r in rewards if int(r.get("policy_expected", 0)) == 0 and "policy_outcome_pct" in r
    ]

    metric: dict[str, float | int] = {
        "n_trials": n_trials,
        "n_completed": len(rewards),
        "n_verifier_failed": n_unscored,
        "completion_rate": (len(rewards) / n_trials) if n_trials else 0.0,
        "n_policy_on": len(on),
        "n_policy_off": len(off),
    }

    def put_mean(name: str, key: str) -> None:
        """Emit mean(`key`) only if some trial reported it — see the module docstring."""
        vals = col(key)
        if vals:
            metric[name] = _mean(vals)

    def put_sum(name: str, key: str) -> None:
        vals = col(key)
        if vals:
            metric[name] = int(sum(vals))

    # outcome — always present (deterministic, every trial)
    put_mean("mean_score", "score")
    put_mean("mean_outcome_correct", "outcome_correct")
    put_mean("mean_outcome_label_parsed", "outcome_label_parsed")

    # patient-evidence grounding + tool-call health (deterministic, in-trial).
    # evidence_calls_error is the load-bearing one: without it, a sweep whose MIMIC calls all
    # failed is indistinguishable from one that simply found little data.
    # mean_grounding_id_precision is averaged ONLY over trials that reported it. A code-mode trial
    # (codex exec scaffold) has no tool-authored observations — its only evidence is the agent's own
    # script stdout — so ID grounding cannot be a fabrication floor there and is deliberately
    # omitted. These two counters make that omission visible instead of leaving the denominator to
    # be inferred from a missing key.
    put_mean("mean_grounding_id_precision", "grounding_id_precision")
    put_sum("sum_grounding_id_untrustworthy_pool", "grounding_id_untrustworthy_pool")
    put_sum("sum_grounding_ids_agent_authored_only", "grounding_ids_agent_authored_only")
    put_sum("sum_grounding_ids_cited", "grounding_ids_cited")
    put_sum("sum_grounding_ids_grounded", "grounding_ids_grounded")
    put_mean("mean_evidence_calls", "evidence_calls")
    put_sum("sum_evidence_calls_ok", "evidence_calls_ok")
    put_sum("sum_evidence_calls_error", "evidence_calls_error")
    # The one to alarm on: auth/permission failures mean the sweep lost data access (cf. the 401
    # job). An ordinary agent-caused failure (calling a tool the sidecar does not serve) is not this.
    put_sum("sum_evidence_calls_auth_error", "evidence_calls_auth_error")
    put_sum("sum_evidence_calls_spilled", "evidence_calls_spilled")
    put_sum("sum_evidence_calls_empty", "evidence_calls_empty")
    put_sum("sum_evidence_calls_unattributed", "evidence_calls_unattributed")
    # >0 ⇒ some trial's id pool is a lower bound (a sub-agent's own tool results are not in the
    # trajectory), so mean_grounding_id_precision under-reports for those trials.
    put_sum("sum_evidence_subagent_summaries", "evidence_subagent_summaries")
    # 0 ⇒ the tool registry failed to import in the container and recognition fell back to the
    # vendored name set. Surfaced so a degraded sweep is visible rather than silently thin.
    put_mean("mean_evidence_registry_loaded", "evidence_registry_loaded")

    # policy grounding — deterministic parts in-trial; support rate only if scored post-hoc
    put_mean("mean_policy_file_recall", "policy_file_recall")
    put_mean("mean_policy_file_precision", "policy_file_precision")
    put_mean("mean_policy_file_f1", "policy_file_f1")
    put_mean("mean_policy_citation_resolve_rate", "policy_citation_resolve_rate")
    put_mean("mean_policy_citation_support_rate", "policy_citation_support_rate")
    # policy_outcome_pct is a mean over the items that could be scored, so it is NOT comparable
    # across trials with different item counts (an unjudged citation_support item raises it by up
    # to 33.3 pts). Surface both so the two can be told apart.
    put_mean("mean_policy_outcome_items_scored", "policy_outcome_items_scored")
    put_mean("mean_policy_support_judged", "policy_support_judged")
    put_sum("sum_policy_citations_total", "policy_citations_total")
    put_sum("sum_policy_citations_fabricated", "policy_citations_fabricated")
    # Complete (support judged) and deterministic-only outcomes are DIFFERENT measurements over
    # different item counts and are never averaged together — see policy_scoring.py.
    on_pct = [float(r["policy_outcome_pct"]) for r in on if "policy_outcome_pct" in r]
    if on_pct:
        metric["mean_policy_outcome_pct_on"] = _mean(on_pct)
        metric["n_policy_outcome_complete"] = len(on_pct)
    det = [
        float(r["policy_outcome_pct_deterministic"])
        for r in on
        if "policy_outcome_pct_deterministic" in r
    ]
    if det:
        metric["mean_policy_outcome_pct_on_deterministic"] = _mean(det)
        metric["n_policy_outcome_deterministic"] = len(det)
    if off:
        metric["mean_policy_outcome_pct_off"] = _mean([float(r["policy_outcome_pct"]) for r in off])

    # process rubric — post-hoc as of step 4; present only when rewards were amended
    put_mean("mean_process_pct", "process_pct")

    # failure flags
    put_sum("sum_judge_failed", "judge_failed")
    put_sum("sum_policy_judge_failed", "policy_judge_failed")
    put_sum("sum_policy_corpus_unavailable", "policy_corpus_unavailable")
    args.output.write_text(json.dumps(metric, indent=2))


if __name__ == "__main__":
    main()
