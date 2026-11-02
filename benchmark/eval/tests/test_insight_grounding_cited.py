"""Tests for what P_claim judges each claim against (insight_grounding_llm.py).

Reviewers asked that the evidence passed to the judge be explicit and that tests cover claims
with incorrect citations, missing citations, and support found only in another retrieved record.
The judge is an LLM, so it is replaced here by a stub that returns "supported" exactly when the
claim's value string occurs in the evidence it was handed. That makes every assertion below a
statement about WHICH EVIDENCE reached the judge, which is the thing under dispute:

  1. pool mode (the published metric): a claim whose citation points at the wrong record is still
     supported when another retrieved record carries the fact; `evidence_refs == "all"`.
  2. cited mode: the same claim is judged only against the cited record and comes back
     unverifiable; `evidence_refs` names that record and nothing else.
  3. cited mode: a claim with no citation handle is `unresolvable`, never sent to the judge.
  4. cited mode: a citation that matches no retrieved record is `unresolved`.
  5. the case's own subject_id / hadm_id are not citations (every row carries them).
  6. a date handle resolves on its calendar day; a tool name resolves to that tool's observations.
  7. score() exposes resolvable_frac and strict_on_resolvable in cited mode only.

    python benchmark/eval/tests/test_insight_grounding_cited.py
"""

from __future__ import annotations

import json
import sys
import types

from clinicare_core.paths import bootstrap

bootstrap(__file__, "eval", "grounding")
import insight_grounding_llm as IG  # noqa: E402
from core.trajectory import Observation  # noqa: E402

_fails: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        _fails.append(label)


# ── stub judge: "supported" iff the claim's number/word token appears in the evidence block ──
_seen_prompts: list[str] = []


def _stub_call(client, model, prompt):
    _seen_prompts.append(prompt)
    ev = prompt.split("EVIDENCE (raw rows/notes the agent retrieved from the database):", 1)[1]
    ev = ev.split("For EACH numbered claim", 1)[0]
    claims_blk = prompt.split("CLAIMS:", 1)[1].split("Return ONLY", 1)[0]
    out = []
    for line in claims_blk.strip().splitlines():
        i, _, text = line.partition(". ")
        # the test claims each carry one distinctive token after "value " — look it up in the evidence
        tok = text.split("value ", 1)[1].split()[0] if "value " in text else text
        out.append({"i": int(i), "verdict": "supported" if tok in ev else "unverifiable", "reason": "stub"})
    return json.dumps(out)


IG._call = _stub_call  # the ONLY LLM touchpoint used by judge_claims / judge_cited

CASE = {"subject_id": {"12014968"}, "hadm_id": {"20449658"}, "stay_id": set()}
OBS = [
    Observation(step_id=2, tool="get_lab_timeline", args={"subject_id": 12014968},
                text='[{"subject_id": 12014968, "hadm_id": 20449658, "charttime": "2180-01-30 15:27:00", "label": "Creatinine", "value 2.3": 1}]'),
    Observation(step_id=3, tool="get_note", args={"note_id": "12014968-RR-9"},
                text='{"note_id": "12014968-RR-9", "subject_id": 12014968, "text": "CT CHEST W/O CONTRAST. value CT_CHEST"}'),
    Observation(step_id=4, tool="get_vitals_timeline", args={"subject_id": 12014968},
                text='[{"subject_id": 12014968, "charttime": "2180-01-29 20:00:00", "label": "HR", "value 112": 1}]'),
]
POOL = "\n".join(o.text for o in OBS)

print("1. pool mode: wrong citation, fact elsewhere in the pool → supported")
claims = ["Creatinine value 2.3 on 2180-01-30 (cited 12014968-RR-9)"]
v = IG.judge_claims(None, "m", claims, POOL)
check("supported against the pool", v[0]["verdict"] == "supported", str(v))
sc = IG.score(claims, [{**v[0], "evidence_refs": "all"}])
check("grounding 1.0, no resolvable_frac field in pool mode", sc["grounding"] == 1.0 and "resolvable_frac" not in sc, str(sc))

print("2. cited mode: the same claim judged only against the cited note → unverifiable")
cc = [{"claim": "Creatinine value 2.3 on 2180-01-30", "cited": ["12014968-RR-9"]}]
_seen_prompts.clear()
v = IG.judge_cited(None, "m", cc, OBS, CASE)
check("unverifiable from the cited record alone", v[0]["verdict"] == "unverifiable", str(v))
check("evidence_refs is exactly the cited note", v[0]["evidence_refs"] == [[3, "get_note"]], str(v[0]["evidence_refs"]))
check("judge saw only the note text", len(_seen_prompts) == 1 and "Creatinine" not in _seen_prompts[0].split("CLAIMS:")[0])

print("3. cited mode: no citation → unresolvable, judge not called")
_seen_prompts.clear()
v = IG.judge_cited(None, "m", [{"claim": "HR value 112", "cited": []}], OBS, CASE)
check("unresolvable", v[0]["verdict"] == "unresolvable", str(v))
check("no judge call", not _seen_prompts)

print("4. cited mode: citation matches nothing retrieved → unresolved")
v = IG.judge_cited(None, "m", [{"claim": "HR value 112", "cited": ["12014968-RR-99"]}], OBS, CASE)
check("unresolved", v[0]["verdict"] == "unresolved" and v[0]["evidence_refs"] == [], str(v))

print("5. the case's own ids are not citations")
v = IG.judge_cited(None, "m", [{"claim": "HR value 112", "cited": ["12014968", "20449658"]}], OBS, CASE)
check("dropped, hence unresolved (handles given but none usable)", v[0]["verdict"] == "unresolved" and v[0]["cited_dropped"] == ["12014968", "20449658"], str(v))

print("6. date handle resolves on its day; tool name resolves to its observations")
idx, used, dropped = IG.resolve_citations(["2180-01-29 20:00"], OBS, CASE)
check("date → vitals row only", idx == [2], str(idx))
idx, used, dropped = IG.resolve_citations(["get_lab_timeline"], OBS, CASE)
check("tool name → lab observation", idx == [0], str(idx))
v = IG.judge_cited(None, "m", [{"claim": "Creatinine value 2.3", "cited": ["get_lab_timeline"]},
                               {"claim": "HR value 112", "cited": ["get_lab_timeline"]}], OBS, CASE)
check("correct citation supported; wrong-tool citation unverifiable",
      [x["verdict"] for x in v] == ["supported", "unverifiable"], str([x["verdict"] for x in v]))

print("6b. identifiers match as complete tokens; table names resolve through the tool map")
OBS2 = OBS + [Observation(step_id=5, tool="get_note", args={"note_id": "12014968-RR-99"},
                          text='{"note_id": "12014968-RR-99", "text": "value 2.3 appears here too"}')]
idx, used, dropped = IG.resolve_citations(["12014968-RR-9"], OBS2, CASE)
check("RR-9 does not select RR-99", idx == [1], str(idx))
idx, used, dropped = IG.resolve_citations(["labevents"], OBS2, CASE)
check("table name → the tool that reads it", idx == [0], str(idx))
idx, used, dropped = IG.resolve_citations(["radiology"], OBS2, CASE)
check("radiology → the note reads (get_note serves radiology notes too)", idx == [1, 3], str(idx))
idx, used, dropped = IG.resolve_citations(["microbiologyevents"], OBS2, CASE)
check("a table no call read → unresolved", idx == [] and dropped == ["microbiologyevents"], str((idx, dropped)))

print("6c. code-mode traces: a tool citation follows the call to its rows")
OBS3 = [Observation(step_id=7, tool="get_lab_timeline", args={"subject_id": "12014968"}, text=""),
        Observation(step_id=7, tool="_exec_output", args={}, text='{"label": "Creatinine", "value 2.3": 1}'),
        Observation(step_id=8, tool="_exec_output", args={}, text='unrelated later output')]
idx, used, dropped = IG.resolve_citations(["get_lab_timeline"], OBS3, CASE)
check("empty-text call pulls in the same step's exec output only", idx == [0, 1], str(idx))
v = IG.judge_cited(None, "m", [{"claim": "Creatinine value 2.3", "cited": ["get_lab_timeline"]}], OBS3, CASE)
check("and the claim is judged supported against those rows", v[0]["verdict"] == "supported", str(v))

print("7. score() derived rates in cited mode")
sc = IG.score([{"claim": "a", "cited": ["x"]}], [{"verdict": "supported"}], mode="cited")
check("fully cited report still reports resolvable_frac 1.0", sc.get("resolvable_frac") == 1.0 and sc.get("strict_on_resolvable") == 1.0, str(sc))
sc = IG.score([{"claim": "a", "cited": []}, {"claim": "b", "cited": ["x"]}, {"claim": "c", "cited": ["y"]}, {"claim": "d", "cited": ["z"]}],
              [{"verdict": "unresolvable"}, {"verdict": "unresolved"}, {"verdict": "supported"}, {"verdict": "contradicted"}])
check("strict grounding counts both failures", sc["grounding"] == 0.25, str(sc["grounding"]))
check("resolvable_frac 3/4", abs(sc["resolvable_frac"] - 0.75) < 1e-9, str(sc))
check("strict_on_resolvable 1/3", abs(sc["strict_on_resolvable"] - 1 / 3) < 1e-9, str(sc))
check("detail keeps the citation handles", sc["detail"][1]["cited"] == ["x"], str(sc["detail"][1]))

print("8. cache identity differs by mode")
k_pool = IG.inputs_key("r", "p", "m", "u")
k_cit = IG.inputs_key("r", "p", "m", "u", mode="cited", structure="s1")
k_cit2 = IG.inputs_key("r", "p", "m", "u", mode="cited", structure="s2")
check("cited key changes when the observation structure changes", k_cit["structure_sha"] != k_cit2["structure_sha"])
check("pool key unchanged shape (no evidence_mode)", "evidence_mode" not in k_pool)
check("cited key tagged and uses the cited prompt", k_cit["evidence_mode"] == "cited" and k_cit["prompts_sha"] != k_pool["prompts_sha"])
check("cache file names differ", IG.cache_name("pool") != IG.cache_name("cited"))

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}"); sys.exit(1)
print("ALL PASS")
