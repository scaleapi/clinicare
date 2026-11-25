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
  8. the cache key differs by evidence mode.
  9. which citation belongs to a claim is decided by rule: the claim's own sentence, else the rest
     of its paragraph / list item / table row; policy citations and the case's own ids are not
     patient-evidence citations; a paragraph break ends the scope.
 9b. a nested list item inherits the citation of the item it sits under.
 9c. the judge is shown the tool call and arguments (e.g. the SQL) with each cited record.
 9d. a code-mode call whose rows arrive in a later output resolves to that output.
 9f. a citation naming several sources resolves to all of them, and a parenthesis cut by the
     sentence splitter is read whole.
 9e. uncertain attachments (paragraph / parent scope, table rows, phrase-only handles) are
     re-audited per handle by the judge model; direct sentence citations are left to the rule.
 10. claims are extracted once per report and reused, so re-judging scores the same claims.

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
check("dropped as case ids → descriptor-only citation → unresolvable", v[0]["verdict"] == "unresolvable" and v[0]["citation_kind"] == "descriptor" and v[0]["cited_dropped"] == ["12014968", "20449658"], str(v))

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
OBS_SQL = OBS2 + [Observation(step_id=6, tool="run_sql", args={"query": "SELECT text FROM discharge_notes WHERE subject_id = 12014968"},
                              text='[{"note_id": "12014968-DS-21", "text": "value DS"}]')]
idx, used, dropped = IG.resolve_citations(["discharge_notes"], OBS_SQL, CASE)
check("SQL table citation resolves through FROM/JOIN + canon (discharge_notes → discharge)", 4 in idx and 1 in idx, str(idx))
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

print("6d. handle shapes reports actually use")
OBS4 = OBS + [Observation(step_id=9, tool="list_orders", args={"subject_id": 12014968},
                          text='[{"poe_id": "12014968-102", "ordertime": "2180-11-12 13:35:00", "order_subtype": "Blood tests"}]')]
for handle, want in [
    ('mimic_get_lab_timeline lab_name="hemoglobin"', [0]),      # tool call snippet → that tool
    ('`list_orders(order_type=["Blood Bank", …])`', [3]),       # tool with args
    ("HADM 24440629", []),                                      # another encounter, not retrieved → unresolved
    ("RR-9", [1]),                                              # short note id, any subject prefix
    ("11-12 13:35", [3]),                                       # month-day time, year elided
    ("POE", [3]),                                               # source word → table → tool
    ("discharge narrative", [1]),                               # source word → discharge table → get_note read it
    ("2180-01-29 20:00 → 21:00", [2]),                          # timestamp range: first stamp's day
]:
    idx, used, dropped = IG.resolve_citations([handle], OBS4, CASE)
    check(f"{handle!r} → {want}", idx == want, str(idx))
idx, used, dropped = IG.resolve_citations(["radiology report 12014968-RR-99"], OBS4, CASE)
check("source word + an id never retrieved → unresolved, not every radiology row", idx == [] and dropped == ["radiology report 12014968-RR-99"], str((idx, dropped)))
idx, used, dropped = IG.resolve_citations(["radiology report 12014968-RR-9"], OBS4, CASE)
check("source word + a retrieved id → that record only", idx == [1], str(idx))
idx, used, dropped = IG.resolve_citations(["get_lab_timeline 2180-01-29 20:00"], OBS4, CASE)
check("tool + timestamp → the timestamp tier (more specific) wins", idx == [2], str(idx))
check("value phrase is a descriptor, not a pointer", IG.citation_kind(["Heparin 5000 units"], CASE) == "descriptor")
check("case id alone is a descriptor", IG.citation_kind(["hadm_id: 20449658"], CASE) == "descriptor")
check("tool snippet is a pointer", IG.citation_kind(['mimic_get_lab_timeline lab_name="x"'], CASE) == "pointer")
v = IG.judge_cited(None, "m", [{"claim": "HR value 112", "cited": ["Heparin 5000 units"]}], OBS4, CASE)
check("descriptor-only citation → unresolvable, not unresolved", v[0]["verdict"] == "unresolvable" and v[0]["citation_kind"] == "descriptor", str(v))

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

print("9. citation attachment by rule")
REPORT = """## Evidence

- CT this admission: radiology note `16706440-RR-41` is a CTA at `2167-03-27 14:17`. The technique states that contrast was given without complication.
- Baseline creatinine was 0.5 mg/dL and KDIGO stage 1 applies [policy: kdigo.md L493-L500]. No later value exceeded it during the stay.

The potassium was 7.3 mEq/L (labevents itemid 50971). Insulin was not charted in emar for this admission.

A separate paragraph makes a claim with no source named anywhere in it at all.

| Lab | Value | Source |
|---|---|---|
| Lactate | 4.1 | `get_lab_timeline` |
"""
U = IG.report_units(REPORT)
texts = [u["text"] for u in U]
at = lambda needle: next(i for i, s in enumerate(texts) if needle in s)  # noqa: E731
check("table separator row is not a unit", not any(set(s) <= set("|-: ") for s in texts), str(texts))
h, scope = IG.attach_citation(U, at("radiology note"))
check("own sentence: note id and timestamp", scope == "sentence" and "16706440-RR-41" in h and "2167-03-27 14:17" in h, str((h, scope)))
h, scope = IG.attach_citation(U, at("The technique states"))
check("no citation in the sentence → inherits the list item's", scope == "block" and "16706440-RR-41" in h, str((h, scope)))
h, scope = IG.attach_citation(U, at("Baseline creatinine"))
check("a policy citation is not a patient-evidence citation", (h, scope) == ([], "none"), str((h, scope)))
h, scope = IG.attach_citation(U, at("potassium was 7.3"))
check("parenthetical naming a table is a citation", scope == "sentence" and h == ["labevents itemid 50971"], str((h, scope)))
h, scope = IG.attach_citation(U, at("Insulin was not charted"))
check("bare MIMIC table name in running text is a citation", scope == "sentence" and h == ["emar"], str((h, scope)))
h, scope = IG.attach_citation(U, at("separate paragraph"))
check("a paragraph break ends the scope", (h, scope) == ([], "none"), str((h, scope)))
h, scope = IG.attach_citation(U, at("Lactate"))
check("table row carries its own source cell", scope == "sentence" and h == ["get_lab_timeline"], str((h, scope)))
check("unit index out of range → no citation", IG.attach_citation(U, -1) == ([], "none") and IG.attach_citation(U, 10**6) == ([], "none"))
check("an ordinary parenthetical is not a citation", IG.unit_handles("Rhythm was SR (Sinus Rhythm) at 19:00 (and V paced later).") == [])
check("ICD-9 / CURB-65 / POD-2 are not note ids", IG.unit_handles("ICD-9 431 (CURB-65 of 0) on POD-2.") == []
      and IG.citation_kind(["ICD-9 431"], CASE) == "descriptor", str(IG.unit_handles("ICD-9 431 (CURB-65 of 0) on POD-2.")))
check("a short note id still is", IG.unit_handles("Per RR-41 the lung was clear enough.") == ["RR-41"])
check("pointer kind: record id > tool/table > date", [IG.pointer_kind(x, CASE) for x in (["16706440-RR-41", "2167-03-27"], ["emar"], ["2167-03-27 14:17"], [])] == ["record", "source", "date", "none"])
check("attachment is deterministic", all(IG.attach_citation(U, i) == IG.attach_citation(IG.report_units(REPORT), i) for i in range(len(U))))

print("9b. nested list items inherit the parent item's citation; siblings and headings do not")
NESTED = """### Imaging

1. **CT Chest with Contrast** (Note: `14563230-RR-28`):
   - **Timestamp**: `2132-03-17 12:11:00`
   - **Location**: Transplant Floor on the general ward
2. **Abdominal ultrasound** with no source named for it
   - **Location**: Radiology suite on the second floor

- **Administration Window**: 07:45 to 08:45 (stay_id 32008189, ICU)
- **Clinical Course**: persistent bleeding via chest tubes overnight
"""
N = IG.report_units(NESTED)
nt = [u["text"] for u in N]
nat = lambda needle: next(i for i, s in enumerate(nt) if needle in s)  # noqa: E731
h, scope = IG.attach_citation(N, nat("Transplant Floor"))
check("sub-bullet with no citation takes the parent item's note id", scope == "parent" and len(h) == 1 and "14563230-RR-28" in h[0], str((h, scope)))
h, scope = IG.attach_citation(N, nat("Timestamp"))
check("sub-bullet with its own citation keeps it", scope == "sentence" and h == ["2132-03-17 12:11:00"], str((h, scope)))
h, scope = IG.attach_citation(N, nat("Radiology suite"))
check("parent without a citation gives none; the earlier item is not inherited", (h, scope) == ([], "none"), str((h, scope)))
h, scope = IG.attach_citation(N, nat("Clinical Course"))
check("a sibling bullet's citation is not inherited", (h, scope) == ([], "none"), str((h, scope)))

print("9c. the judge sees which call produced a cited record")
SQL = Observation(step_id=4, tool="run_sql", args={"query": "SELECT COUNT(*) FROM procedureevents WHERE hadm_id = 1"},
                  text='{"rows": [{"count_star()": 0}]}')
_seen_prompts.clear()
IG.judge_cited(None, "m", [{"claim": "value procedureevents empty", "cited": ["run_sql"]}], [SQL], CASE)
check("the SQL text reaches the judge with its result", "FROM procedureevents" in _seen_prompts[-1] and "count_star" in _seen_prompts[-1], _seen_prompts[-1][-300:])

print("9d. an async exec call resolves to the later output that carries its rows")
ASYNC = [
    Observation(step_id=11, tool="get_ed_visit_details", args={"stay_id": 30252376}, text=""),
    Observation(step_id=11, tool=IG.EXEC_OUTPUT_TOOL, args={}, text="Script running with cell ID 5\nWall time 10.0 seconds\nOutput:\n"),
    Observation(step_id=12, tool=IG.EXEC_OUTPUT_TOOL, args={}, text="Script completed\nOutput:\n{\"triage\": {\"resprate\": 22}}"),
    Observation(step_id=13, tool=IG.EXEC_OUTPUT_TOOL, args={}, text="Script completed\nOutput:\nunrelated later cell"),
]
idx, used, dropped = IG.resolve_citations(["get_ed_visit_details"], ASYNC, CASE)
check("stub plus the completing output, not the cell after it", idx == [0, 1, 2], str(idx))
SYNC = [ASYNC[0], Observation(step_id=11, tool=IG.EXEC_OUTPUT_TOOL, args={}, text="Script completed\nOutput:\nrows"), ASYNC[3]]
idx, _, _ = IG.resolve_citations(["get_ed_visit_details"], SYNC, CASE)
check("a sync call still resolves to its own step's output only", idx == [0, 1], str(idx))

print("9e. uncertain attachments are re-audited per handle; direct ones are left to the rule")
ROW = "| 9 | UA with culture | UA at 2147-11-11 18:56:00 showed RBC-6. Urine culture NO GROWTH [microbiologyevents]. |"
AUD = IG.report_units("Potassium was 7.3 (labevents itemid 50971).\n\n" + ROW + "\n\n- Per `16706440-RR-41` the lung was clear. The drain was placed as planned.\n")
ai = lambda needle: next(i for i, u in enumerate(AUD) if needle in u["text"])  # noqa: E731
aclaims = [{"claim": "K was 7.3", "unit": ai("Potassium")}, {"claim": "UA showed RBC-6", "unit": ai("RBC-6")},
           {"claim": "The drain was placed as planned", "unit": ai("drain was placed")}, {"claim": "uncited", "unit": -1}]
for c in aclaims:
    c["cited"], c["cited_scope"] = IG.attach_citation(AUD, c["unit"])
check("direct sentence citation is not uncertain", not IG.audit_needed(aclaims[0], AUD))
check("table row and paragraph fallback are uncertain", IG.audit_needed(aclaims[1], AUD) and IG.audit_needed(aclaims[2], AUD))
check("an uncited claim is never audited", not IG.audit_needed(aclaims[3], AUD))
check("a phrase naming a source type is uncertain", IG.audit_needed({"cited": ["only 1 discharge note and 7 radiology notes"], "cited_scope": "sentence", "unit": ai("Potassium")}, AUD))
_real_call2 = IG._call
_audit_prompts: list[str] = []


def _audit_stub(client, model, prompt):
    _audit_prompts.append(prompt)
    # claim 1 (the UA): keep the timestamp handle only; claim 2 (the drain): keep nothing
    ua_handles = aclaims[1]["cited"]
    ts = next(j for j, h in enumerate(ua_handles, 1) if h.startswith("2147"))
    return json.dumps([{"i": 1, "keep": [ts], "reason": "time of the UA"}, {"i": 2, "keep": [], "reason": "belongs to the lung finding"}])


IG._call = _audit_stub
keep = IG.audit_citations(None, "m", aclaims, AUD)
IG._call = _real_call2
check("only the uncertain claims were sent", "K was 7.3" not in _audit_prompts[-1] and "UA showed RBC-6" in _audit_prompts[-1] and "drain was placed" in _audit_prompts[-1])
IG.apply_audit(aclaims, AUD, keep)
check("direct claim untouched", aclaims[0]["audit"] == "direct" and aclaims[0]["cited"] == aclaims[0]["cited_rule"], str(aclaims[0]))
check("per-handle: the UA keeps its timestamp and loses the culture's table", aclaims[1]["audit"] == "partial" and aclaims[1]["cited"] == ["2147-11-11 18:56"]
      and "microbiologyevents" in aclaims[1]["cited_rule"], str(aclaims[1]))
check("a claim whose handles all belong elsewhere becomes uncited", aclaims[2]["audit"] == "dropped" and aclaims[2]["cited"] == [] and aclaims[2]["cited_rule"], str(aclaims[2]))
v = IG.judge_cited(None, "m", [aclaims[2]], OBS4, CASE)
check("…and is scored as having no citation, not as a wrong one", v[0]["verdict"] == "unresolvable" and v[0]["citation_kind"] == "none", str(v))
c_un = [{"claim": "x", "unit": ai("drain was placed"), "cited": ["16706440-RR-41"], "cited_scope": "block"}]
IG.apply_audit(c_un, AUD, {})
check("no answer from the model leaves the rule's citation in place", c_un[0]["audit"] == "unanswered" and c_un[0]["cited"] == ["16706440-RR-41"])

print("9f. a citation that names several sources resolves to each of them")
MULTI = [
    Observation(step_id=1, tool="list_orders", args={"subject_id": 15167397}, text='{"ordertime": "2172-01-29 08:48:32", "order_subtype": "Nuclear Med"}'),
    Observation(step_id=2, tool="get_note", args={"note_id": "15167397-DS-11"}, text='{"note_id": "15167397-DS-11", "text": "HIDA scan: patent common bile duct"}'),
    Observation(step_id=3, tool="get_imaging_studies", args={"subject_id": 15167397}, text='{"items": []}'),
    Observation(step_id=4, tool="get_data_availability", args={"subject_id": 15167397}, text='{"stream": "radiology_notes", "rows": 0}'),
    Observation(step_id=5, tool="run_sql", args={"query": "SELECT note_id FROM radiology_notes WHERE subject_id=15167397"}, text='{"rows": []}'),
    Observation(step_id=6, tool="get_lab_timeline", args={"subject_id": 15167397}, text='{"charttime": "2172-02-02 05:00:00", "lab_name": "Lactate"}'),
]
idx, used, dropped = IG.resolve_citations(["ordered 2172-01-29 08:48, `list_orders`; result transcribed in DS-11"], MULTI, CASE)
check("';' separates two citations: the order row and the note both reach the judge", idx == [0, 1], str(idx))
idx, _, _ = IG.resolve_citations(["radiology report 15167397-DS-11"], MULTI, CASE)
check("within one part the most specific tier still decides (note id, not every radiology row)", idx == [1], str(idx))
idx, _, _ = IG.resolve_citations(["`get_imaging_studies`, `get_data_availability`, direct SQL on `radiology_notes`"], MULTI, CASE)
check("every tool named is resolved, and 'SQL' is run_sql", idx == [2, 3, 4], str(idx))
idx, _, _ = IG.resolve_citations(["mimic_get_imaging_studies; mimic_get_lab_timeline"], MULTI, CASE)
check("two tools separated by ';'", idx == [2, 5], str(idx))

CUT = (
    "- No PE-directed testing is documented. That report's findings were cardiomegaly and mild fluid overload, "
    "with no PE-directed imaging or PE result (`get_imaging_studies`; `get_note` on `15991401-RR-13`). "
    "A later paragraph sentence stands alone without any source at all.\n"
)
units = IG.report_units(CUT)
cut_i = next(k for k, u in enumerate(units) if "cardiomegaly" in u["text"])
check("the splitter cuts the parenthesis at ';' (the case this guards)", "RR-13" not in units[cut_i]["text"], units[cut_i]["text"])
check("the cut unit and its continuation are one sentence", IG.sentence_run(units, cut_i) == [cut_i, cut_i + 1], str(IG.sentence_run(units, cut_i)))
hs, scope = IG.attach_citation(units, cut_i)
check("the claim gets the whole parenthesis, at sentence scope",
      scope == "sentence" and any("RR-13" in h and "get_imaging_studies" in h for h in hs), f"{scope} {hs}")
STRAY = "- Dose given 1) at 08:00 and 2) at 10:00 (see below. Next sentence is long enough to stand alone here. And a third sentence with `get_note` in it.\n"
su = IG.report_units(STRAY)
check("a parenthesis that never closes joins nothing", IG.sentence_run(su, 0) == [0], str(IG.sentence_run(su, 0)))

print("10. claims are extracted once and reused")
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

_real_call, _n_extract = IG._call, [0]


def _extract_stub(client, model, prompt):
    _n_extract[0] += 1
    return json.dumps([{"claim": f"draw {_n_extract[0]}", "unit": 2}, {"claim": "bad unit", "unit": "x"}])


IG._call = _extract_stub
with tempfile.TemporaryDirectory() as td:
    cache = Path(td) / IG.CLAIMS_CACHE_NAME
    c1 = IG.load_or_extract_claims(None, "m", REPORT, "u", cache)
    c2 = IG.load_or_extract_claims(None, "m", REPORT, "u", cache)
    check("second call reads the cache, no new extraction", _n_extract[0] == 1 and c1 == c2, str((_n_extract[0], c1, c2)))
    check("unit numbers are stored zero-based; unusable ones as -1", [c["unit"] for c in c1] == [1, -1], str(c1))
    c3 = IG.load_or_extract_claims(None, "m", REPORT + " edited", "u", cache)
    check("an edited report is re-extracted", _n_extract[0] == 2 and c3 != c1, str(_n_extract[0]))
IG._call = _real_call

sc = IG.score(
    [{"claim": "a", "cited": ["x"], "cited_scope": "sentence"}, {"claim": "b", "cited": ["y"], "cited_scope": "block"},
     {"claim": "c", "cited": [], "cited_scope": "none"}, {"claim": "d", "cited": ["z"], "cited_scope": "sentence"}],
    [{"verdict": "supported", "citation_kind": "pointer"}, {"verdict": "unverifiable", "citation_kind": "pointer"},
     {"verdict": "unresolvable", "citation_kind": "none"}, {"verdict": "unresolved", "citation_kind": "pointer"}], mode="cited")
check("coverage counts every claim with a citation, matched or not", abs(sc["coverage"] - 0.75) < 1e-9, str(sc["coverage"]))
check("citation precision is over cited claims only", abs(sc["citation_precision"] - 1 / 3) < 1e-9, str(sc["citation_precision"]))
check("sentence-only pair", abs(sc["coverage_sentence"] - 0.5) < 1e-9 and abs(sc["citation_precision_sentence"] - 0.5) < 1e-9, str(sc))

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}"); sys.exit(1)
print("ALL PASS")
