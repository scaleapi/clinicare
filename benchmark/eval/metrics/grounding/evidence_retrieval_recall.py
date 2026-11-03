"""evidence_retrieval_recall — retrieval coverage (R_retrieval in the paper).

Did the agent read the source tables the answer requires?

Adapted from DRBench's `retrieval_recall` (set overlap of retrieved vs gold sources) to
CliniCARE-Bench at **source-table granularity** — the granularity the clinician annotation gives us.

  recall = |gold ∩ retrieved| / |gold|      (per trial; also precision / F1)

- gold      : per-scenario required MIMIC tables, from the clinician annotation delivery CSV
              (grounding_requirements; the edited version when present). Table-level,
              expert-validated — needs NO live MIMIC.
- retrieved : the tables the agent's MIMIC tool calls actually read, from the trace.
              Tool→table map is vendored from packages/mimic_tools tool_registry.TOOLS
              (each tool declares the logical tables it reads); run_sql tables are
              parsed from the query's FROM/JOIN clauses.

Everything is trace-only + deterministic (no LLM, no DB). The scenario→annotation join is on
the task NUMBER, which both sides carry (see core.scenarios); the resolved titles are logged.

Scope. A required source counts as consulted only when the call that read it was scoped to the
case's patient: a call whose ``subject_id`` argument (or ``subject_id = N`` in a run_sql query)
names a different patient credits nothing, and the trial's ``scope`` block counts such calls.
Calls that name no subject_id (``get_admission_details(hadm_id=...)``, ``get_note(note_id=...)``)
cannot be checked without the database and are credited as before; they are counted as
``unscoped``. Encounter scope is REPORTED, not enforced: a call to another admission of the same
patient (``hadm_id`` differs from the one the task names) is legitimate in the multi-admission
scenarios (baseline creatinine, frequent-admitter, renal trajectory), so it is counted under
``other_encounter`` and still credited. Temporal scope is not checked. ``--no-scope`` reproduces
the published numbers, which applied none of this.

Gold source: a committed derived JSON (build_gold.py; per scenario: title + gold source
tables only — no clinical text). The raw annotation CSV stays local / gitignored.

  cd <repo root>
  python benchmark/eval/metrics/grounding/evidence_retrieval_recall.py \
      --traces benchmark/eval/traces \
      --gold   benchmark/eval/gold/gold_sources.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from clinicare_core.paths import bootstrap

HERE = Path(__file__).resolve().parent

bootstrap(__file__, "eval")
from core.scenarios import slug_from_trial_dir, task_by_slug, task_from_title  # noqa: E402
from core.trajectory import (  # noqa: E402
    Observation,
    call_scope,
    case_scope,
    load_trajectory,
    subagent_observations,
)

# ── tool → logical tables (vendored from mimic_tools/tool_registry.py::TOOLS[*]["tables"];
#    re-sync if the registry changes). run_sql is empty → parsed from its query. ──────────
TOOL_TABLES: dict[str, list[str]] = {
    "get_patient_demographics": ["patients", "admissions", "edstays", "icustays"],
    "list_admissions": ["admissions", "diagnoses_icd", "d_icd_diagnoses"],
    "get_admission_details": [
        "admissions",
        "edstays",
        "transfers",
        "icustays",
        "services",
        "drgcodes",
    ],
    "list_ed_visits": ["edstays"],
    "get_ed_visit_details": ["edstays", "ed_triage", "ed_vitalsign", "ed_diagnosis", "ed_pyxis"],
    "get_medication_reconciliation": ["medrecon", "edstays"],
    "list_icu_stays": ["icustays"],
    "get_icu_stay_details": [
        "icustays",
        "chartevents",
        "procedureevents",
        "inputevents",
        "outputevents",
        "labevents",
        "emar",
    ],
    "get_lab_timeline": ["labevents", "d_labitems"],
    "list_notes": ["discharge_notes", "radiology_notes"],
    "get_note": ["discharge_notes", "radiology_notes"],
    "search_notes": ["discharge_notes", "radiology_notes"],
    "get_vitals_timeline": ["chartevents", "d_items"],
    "get_chart_timeline": ["chartevents", "d_items"],
    "get_medication_administered": ["emar", "emar_detail"],
    "get_medication_prescribed": ["prescriptions"],
    "get_diagnosis_history": [
        "diagnoses_icd",
        "d_icd_diagnoses",
        "admissions",
        "ed_diagnosis",
        "edstays",
    ],
    "get_procedures": ["procedures_icd", "d_icd_procedures", "procedureevents", "d_items"],
    "get_imaging_studies": ["radiology_notes", "radiology_detail"],
    "get_microbiology_results": ["microbiologyevents"],
    "get_fluid_balance": ["inputevents", "outputevents", "d_items"],
    "get_icu_administrations": ["inputevents", "d_items"],
    "list_orders": ["poe", "poe_detail"],
    # DELIBERATELY [] even though the registry declares 16 tables for it: this tool returns
    # row COUNTS and time spans, never clinical content. Crediting its tables would let one
    # call score full evidence-retrieval recall without reading a single patient value.
    "get_data_availability": [],
    "run_sql": [],
}

# Physical MIMIC-IV table vocabulary — for extracting gold tables from the free-text
# grounding_requirements (matched as whole words; incl. the annotation's shorthand labs/notes).
GOLD_VOCAB = [
    "admissions",
    "patients",
    "transfers",
    "diagnoses_icd",
    "procedures_icd",
    "drgcodes",
    "emar",
    "emar_detail",
    "hcpcsevents",
    "labevents",
    "microbiologyevents",
    "omr",
    "poe",
    "poe_detail",
    "prescriptions",
    "pharmacy",
    "services",
    "icustays",
    "chartevents",
    "inputevents",
    "outputevents",
    "procedureevents",
    "datetimeevents",
    "ingredientevents",
    "edstays",
    "triage",
    "medrecon",
    "pyxis",
    "vitalsign",
    "radiology",
    "radiology_detail",
    "discharge",
    "discharge_detail",
    "labs",
    "notes",
]

# Lookup/dictionary tables — never "evidence"; dropped from both sides so they don't
# distort precision (gold never lists them).
_DROP = {"d_labitems", "d_items", "d_icd_diagnoses", "d_icd_procedures"}

# Canonicalize logical/shorthand names → the physical name the gold uses. A token maps to
# a SET: normally one physical name; the "notes" shorthand fans out to both note tables;
# an empty set drops the token. Unlisted → itself. Every non-empty target here must also be
# reachable in gold (i.e. in GOLD_VOCAB) — otherwise the prediction token is dead.
_CANON: dict[str, set[str]] = {
    "discharge_notes": {"discharge"},
    "discharge_detail": {"discharge"},
    "radiology_notes": {"radiology"},
    "radiology_detail": {"radiology"},
    "ed_triage": {"triage"},
    "ed_vitalsign": {"vitalsign"},
    "ed_pyxis": {"pyxis"},
    # ed_diagnosis (ED module's diagnosis table) → dropped: its physical name is the bare
    # word "diagnosis", too generic to match safely in the gold's free-text grounding
    # requirements, so gold can never list it and the token would be dead. Note this is
    # NOT diagnoses_icd (inpatient) — mapping it there would let get_ed_visit_details
    # wrongly satisfy a diagnoses_icd requirement. Inpatient diagnoses_icd is still
    # attributed via get_diagnosis_history's own diagnoses_icd entry.
    "ed_diagnosis": set(),
    "emar_detail": {"emar"},
    "poe_detail": {"poe"},
    "labs": {"labevents"},
    "notes": {"radiology", "discharge"},
}


def canon(tables) -> set[str]:
    out: set[str] = set()
    for t in tables:
        t = t.lower()
        if t in _DROP:
            continue
        out |= _CANON.get(t, {t})
    return out


# ── gold: per-scenario required source tables ────────────────────────────────────────────────
# Committed as a derived JSON (build_gold.py) — [{"title": ..., "tables": [...]}] — so the
# repo carries only task title + gold source-table names, NOT the raw annotation (which has
# MIMIC-note-derived clinical text). The raw annotation CSV stays local / gitignored.
def load_gold(gold_json: Path) -> list[dict]:
    """Load the derived gold: [{title, tables(set)}]."""
    data = json.loads(Path(gold_json).read_text())
    return [{"title": g["title"], "tables": set(g["tables"])} for g in data]


def match_scenario(trial: str, gold: list[dict]) -> dict | None:
    """The annotation row for a trial dir, joined on the TASK NUMBER.

    Both sides carry it: a gold title is ``Task N — ...`` and the scenario's ``scenario-NN-`` dir
    prefix gives the same N (see core.scenarios). Matching on title text instead is unsound — a
    scenario id can share no token with its task title — and a wrong-but-plausible match scores a
    trial against another scenario's required tables with nothing to show for it.
    """
    slug = slug_from_trial_dir(trial)
    if slug is None:
        return None
    want = task_by_slug().get(slug)
    return next((g for g in gold if task_from_title(g["title"]) == want), None)


# ── prediction: tables the agent actually read, from the trajectory ──────────────────────
_FROM_RX = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", re.I)
# Real table names we recognize — used to drop run_sql FROM/JOIN hits that are actually CTE
# names, query aliases, or `information_schema` introspection rather than MIMIC tables.
_RECOGNIZED = set(GOLD_VOCAB) | set(_CANON) | {t for tbls in TOOL_TABLES.values() for t in tbls}


def _scope_stats() -> dict:
    return {"case_subject_id": None, "case_hadm_id": None, "calls": 0, "in_scope": 0,
            "unscoped": 0, "wrong_patient": 0, "other_encounter": 0}


def scope_check(obs: Observation, case: dict[str, set[str]], stats: dict) -> bool:
    """True when `obs` may credit tables: its subject_id (if it names one) is the case's patient.
    Updates `stats`. With no case subject_id (instruction names none) every call is credited."""
    stats["calls"] += 1
    cs = call_scope(obs)
    case_subj, case_hadm = case.get("subject_id") or set(), case.get("hadm_id") or set()
    if case_hadm and cs["hadm_id"] and not cs["hadm_id"] <= case_hadm:
        stats["other_encounter"] += 1
    if not case_subj or not cs["subject_id"]:
        stats["unscoped"] += 1
        return True
    if cs["subject_id"] <= case_subj:
        stats["in_scope"] += 1
        return True
    stats["wrong_patient"] += 1
    return False


def tables_from_evidence(evidence_file: Path, case: dict | None = None, stats: dict | None = None) -> set[str] | None:
    """Tables read, from the in-trial ``evidence.jsonl`` (verifier/evidence.jsonl).

     Preferred over re-deriving from the trajectory: evidence.py extracted the CALLS in the
     container from the native trajectory, so there is no per-harness parsing to get wrong here.

     The tool→table mapping and ``canon`` folding happen HERE, not there. Both are tied to the GOLD
     vocabulary and revisable, so the artifact stores only `tool` + `args`
    . In particular this re-applies the
     ``get_data_availability -> []`` override in TOOL_TABLES above: the live registry declares 16
     tables for it, and crediting them let a single row-count call score ~0.87 mean recall.

     Returns None when the file is absent, or when it contains no record whose tables can be
     attributed — the caller then falls back to trajectory parsing rather than scoring 0.
    """
    if not evidence_file.is_file():
        return None
    raw: set[str] = set()
    attributable = 0
    for line in evidence_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        # `ok` = the call returned rows. `unattributed` = a code-mode call recovered from an exec
        # script: the script proves the call was made even though its rows come back in a separate
        # exec_output record, so it counts as consulted. Keeping only `ok` dropped EVERY code-mode
        # record and returned an empty set, scoring those systems 0.00 with no fallback.
        # `spilled` = the call succeeded but the harness elided the payload and we could not read it
        # back (post-hoc without agent/sessions/). Table attribution comes from the TOOL NAME, so it
        # is still valid — excluding it would drop a real retrieval.
        # Per the evaluation spec (§Patient-Evidence Grounding): "A source counts as consulted when
        # a correctly scoped query is issued, whether it returns records or a VALID EMPTY RESULT,
        # because an empty result may itself establish that the evidence is unavailable." Excluding
        # empties under-credits exactly the agents reasoning correctly on lack-of-data cases.
        # A valid empty result (n_rows == 0) counts; a blank/absent payload (n_rows is None) does
        # not — we received nothing, which is not evidence that nothing exists.
        st = rec.get("status")
        if st == "empty" and rec.get("n_rows") != 0:
            continue
        if st not in ("ok", "unattributed", "spilled", "empty"):
            continue
        if rec.get("kind") not in ("mimic", None):
            continue  # policy / exec_output / subagent records read no MIMIC table
        tool = rec.get("tool") or ""
        if case is not None and tool in TOOL_TABLES and not scope_check(
            Observation(step_id=None, tool=tool, args=rec.get("args") or {}, text=""), case, stats
        ):
            continue  # scoped to another patient: credits nothing
        if tool == "run_sql":
            q = str(
                (rec.get("args") or {}).get("query") or (rec.get("args") or {}).get("sql") or ""
            )
            hits = {m.group(1).lower() for m in _FROM_RX.finditer(q)} & _RECOGNIZED
            raw |= hits
            attributable += 1
        elif tool in TOOL_TABLES:
            raw |= set(TOOL_TABLES[tool])
            attributable += 1
    if not attributable:
        return None
    return canon(raw)


def retrieved_tables(observations, case: dict | None = None, stats: dict | None = None) -> set[str]:
    raw: set[str] = set()
    for obs in observations:
        if case is not None and obs.tool in TOOL_TABLES and not scope_check(obs, case, stats):
            continue  # scoped to another patient: credits nothing
        if obs.tool == "run_sql":
            q = obs.args.get("query", "") if isinstance(obs.args, dict) else ""
            # keep only recognized MIMIC tables — filters CTE/alias/information_schema noise
            raw |= {m.group(1).lower() for m in _FROM_RX.finditer(q)} & _RECOGNIZED
        elif obs.tool in TOOL_TABLES:
            raw |= set(TOOL_TABLES[obs.tool])
        # obs.tool == "_subagent" (or unknown): no reliable table attribution → skip
    return canon(raw)


def score(gold_tables: set[str], pred_tables: set[str]) -> dict:
    hit = gold_tables & pred_tables
    recall = len(hit) / len(gold_tables) if gold_tables else None
    precision = len(hit) / len(pred_tables) if pred_tables else None
    # None (not 0.0) when either component is undefined, consistent with recall/precision.
    if recall is None or precision is None:
        f1 = None
    elif recall + precision == 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 3)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "gold": sorted(gold_tables),
        "retrieved": sorted(pred_tables),
        "missed": sorted(gold_tables - pred_tables),
        "extra": sorted(pred_tables - gold_tables),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", type=Path, default=HERE.parents[1] / "traces")
    ap.add_argument(
        "--gold",
        type=Path,
        default=HERE.parents[1] / "gold" / "gold_sources.json",
        help="derived gold JSON (build with build_gold.py)",
    )
    ap.add_argument("--verbose", action="store_true", help="print gold/retrieved/missed per trial")
    ap.add_argument(
        "--no-scope",
        action="store_true",
        help="credit every call regardless of the patient it names (the published behaviour)",
    )
    ap.add_argument("--out", type=Path, default=None, help="write per-trial scores JSON here")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    trials = sorted(args.traces.glob("*/*/*/trajectory.json"))
    if not trials:
        sys.exit(f"no trajectories under {args.traces}")

    seen_match: dict[str, str] = {}
    rows, recalls = [], []
    for tj in trials:
        run, slug = tj.parts[-4], tj.parts[-3]
        g = match_scenario(slug, gold)
        if g is None:
            print(f"  ! {slug}: no annotation row for this scenario", file=sys.stderr)
            continue
        seen_match[slug] = g["title"][:48]
        traj = load_trajectory(tj)
        obs = list(traj.observations)
        # A sub-agent hides its MIMIC calls in agent/sessions/; recover them from the
        # subagents/*.jsonl if fetched. If the agent used a sub-agent but those files
        # weren't fetched, retrieval is incomplete → NA (re-fetch to score).
        sub = subagent_observations(tj.parent) if traj.used_subagent else []
        obs += sub
        # Prefer the normalized in-trial artifact; fall back to trajectory parsing for traces
        # predating evidence.py.
        ev = tj.parents[1] / "verifier" / "evidence.jsonl"
        case = None if args.no_scope else case_scope(traj.steps)
        stats = _scope_stats()
        if case is not None:
            stats["case_subject_id"] = sorted(case["subject_id"])
            stats["case_hadm_id"] = sorted(case["hadm_id"])
        pred = tables_from_evidence(ev, case, stats)
        if pred is None:
            stats = _scope_stats() if case is None else {**_scope_stats(), "case_subject_id": stats["case_subject_id"], "case_hadm_id": stats["case_hadm_id"]}
            pred_traj = retrieved_tables(obs, case, stats)
        s = score(g["tables"], pred if pred is not None else pred_traj)
        s["scope"] = stats
        s["source"] = "evidence.jsonl" if pred is not None else "trajectory"
        s["subagent"] = traj.used_subagent
        s["subagent_recovered"] = bool(sub)
        # NA regardless of source when a sub-agent's calls weren't recovered: evidence.jsonl carries
        # only the parent-visible SUMMARY of a sub-agent, never its own tool results, so the pool is
        # knowingly incomplete and a low recall would be an artifact.
        s["na"] = traj.used_subagent and not sub
        rows.append((run, slug, traj.agent, s))
        if s["recall"] is not None and not s["na"]:
            recalls.append(s["recall"])

    print("=== scenario → annotation join (by task number) ===")
    for slug, m in sorted(seen_match.items()):
        print(f"  {slug:20s} -> {m}")

    print(f"\n{'run':30s} {'scenario':18s} {'recall':>6s} {'missed tables'}")
    for run, slug, agent, s in rows:
        rec = "  NA" if s["na"] else (f"{s['recall']:.2f}" if s["recall"] is not None else "  -")
        flag = ""
        if s["subagent_recovered"]:
            flag = " [subagent calls recovered from sessions]"
        elif s["na"]:
            flag = " [subagent: sessions not fetched → re-fetch to score]"
        print(f"  {run[:28]:28s} {slug:18s} {rec:>6s} {s['missed'] or ''}{flag}")
        if args.verbose:
            print(f"      gold={s['gold']}\n      got ={s['retrieved']}")

    if recalls:
        print(f"\nmean recall over {len(recalls)} trials: {sum(recalls)/len(recalls):.3f}")
    if not args.no_scope and rows:
        tot = {k: sum(s["scope"][k] for _, _, _, s in rows) for k in ("calls", "in_scope", "unscoped", "wrong_patient", "other_encounter")}
        flagged = sum(1 for _, _, _, s in rows if s["scope"]["wrong_patient"])
        print(
            f"scope: {tot['calls']} MIMIC calls — {tot['in_scope']} on the case patient, "
            f"{tot['unscoped']} name no subject_id, {tot['wrong_patient']} on another patient "
            f"(in {flagged} trials); {tot['other_encounter']} name another encounter of the patient"
        )
    if args.out:
        args.out.write_text(json.dumps(
            [{"run": run, "scenario": slug, "agent": agent, **s} for run, slug, agent, s in rows], indent=1
        ) + "\n")
        print(f"wrote per-trial scores → {args.out}")

    # Found inputs but scored none => every one failed to join (missing report, no gold match).
    # Exit non-zero rather than printing an empty table and exiting 0: a silent 0/N reads as
    # "nothing to score" when it means the join is broken.
    if not recalls:
        print(
            f"\n❌ scored 0 of {len(trials)} trace(s) — nothing joined",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
