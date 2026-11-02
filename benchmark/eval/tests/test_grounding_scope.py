"""Tests for the scope checks behind R_retrieval and the bash code-mode trajectory parser.

What the paper claims for retrieval coverage is "queried with the correct patient and, where
applicable, encounter and temporal scope". These tests pin down exactly what the code checks:

  1. a required table read for ANOTHER patient credits nothing (patient scope enforced);
  2. a read of another admission of the SAME patient is credited but counted as `other_encounter`
     (encounter scope reported, not enforced — multi-admission scenarios need those reads);
  3. a call that names no subject_id (get_note by note_id) is credited and counted `unscoped`;
  4. run_sql scope is read from the query text (`subject_id = N`, `IN (...)`);
  5. the case ids come from the task instruction in both phrasings the scenarios use;
  6. mini-swe-agent's bash code mode: tool calls inside a Python heredoc are recovered with their
     scope ids, a failed command credits nothing, and the command's stdout (head+tail when the
     harness elided the middle) enters the evidence pool.

Synthetic fixtures only — no job dirs, no MIMIC, no LLM.

    python benchmark/eval/tests/test_grounding_scope.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from clinicare_core.paths import bootstrap

bootstrap(__file__, "eval", "grounding")
from core.trajectory import (  # noqa: E402
    EXEC_OUTPUT_TOOL,
    Observation,
    call_scope,
    case_scope,
    load_trajectory,
)
from evidence_retrieval_recall import _scope_stats, retrieved_tables, score  # noqa: E402

_fails: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        _fails.append(label)


CASE = {"subject_id": {"12014968"}, "hadm_id": {"20449658"}, "stay_id": set()}
GOLD = {"labevents", "icustays", "radiology"}


def obs(tool, args):
    return Observation(step_id=1, tool=tool, args=args, text="")


print("1. another patient's read credits nothing")
st = _scope_stats()
pred = retrieved_tables([obs("get_lab_timeline", {"subject_id": 99999999})], CASE, st)
check("labevents not credited", "labevents" not in pred, str(pred))
check("counted as wrong_patient", st["wrong_patient"] == 1 and st["in_scope"] == 0, str(st))
s = score(GOLD, pred)
check("recall is 0 for that trial", s["recall"] == 0.0, str(s["recall"]))

print("2. another admission of the same patient is credited, counted other_encounter")
st = _scope_stats()
pred = retrieved_tables([obs("get_lab_timeline", {"subject_id": 12014968, "hadm_id": 11111111})], CASE, st)
check("labevents credited", "labevents" in pred, str(pred))
check("in_scope + other_encounter", st["in_scope"] == 1 and st["other_encounter"] == 1, str(st))

print("3. a call naming no subject_id is credited and counted unscoped")
st = _scope_stats()
pred = retrieved_tables([obs("get_note", {"note_id": "12014968-RR-9"})], CASE, st)
check("radiology credited", "radiology" in pred, str(pred))
check("counted unscoped", st["unscoped"] == 1, str(st))

print("4. run_sql scope is read from the query")
cs = call_scope(obs("run_sql", {"query": "SELECT * FROM labevents WHERE subject_id = 99999999"}))
check("subject_id parsed from WHERE", cs["subject_id"] == {"99999999"}, str(cs))
st = _scope_stats()
pred = retrieved_tables([obs("run_sql", {"query": "SELECT * FROM labevents WHERE subject_id = 99999999"})], CASE, st)
check("wrong-patient SQL credits nothing", "labevents" not in pred and st["wrong_patient"] == 1, str((pred, st)))
cs = call_scope(obs("run_sql", {"query": "select itemid from labevents where hadm_id in (20449658, 20449659)"}))
check("IN (...) parsed", cs["hadm_id"] == {"20449658", "20449659"}, str(cs))
st = _scope_stats()
pred = retrieved_tables([obs("run_sql", {"query": "SELECT * FROM labevents WHERE subject_id = 12014968"})], CASE, st)
check("in-scope SQL credited", "labevents" in pred and st["in_scope"] == 1, str((pred, st)))

print("5. case ids come from the instruction, both phrasings")
msg_eq = "For the ICU stay subject_id=12014968, hadm_id=20449658, did this patient ...\n\n---\n\n## env (ids here are examples: subject_id=10001217)"
msg_sp = "For subject_id 16474066, with multiple admissions, is renal function worsening?\n\n---\n"
c1 = case_scope([{"source": "user", "message": msg_eq}])
c2 = case_scope([{"source": "user", "message": msg_sp}, {"source": "agent", "message": "subject_id=5"}])
check("subject_id= form", c1["subject_id"] == {"12014968"} and c1["hadm_id"] == {"20449658"}, str(c1))
check("boilerplate ids after --- ignored", "10001217" not in c1["subject_id"], str(c1))
check("'subject_id N' form, no hadm", c2["subject_id"] == {"16474066"} and not c2["hadm_id"], str(c2))
check("no instruction → empty scope, every call credited",
      retrieved_tables([obs("get_lab_timeline", {"subject_id": 1})], case_scope([]), _scope_stats()) == {"labevents"})

print("6. bash code mode (mini-swe-agent)")
script = (
    "cd /workspace && python3 <<'PYEOF'\nfrom mimic_tools import mcp_server\n"
    "r = mcp_server._run_tool(\"list_icu_stays\", {\"subject_id\": 12014968, \"hadm_id\": 20449658})\n"
    "print(\"next: get_lab_timeline\")\nfor x in r: print(x.text)\nPYEOF\n"
)
rows = json.dumps([{"stay_id": 31929757, "subject_id": 12014968, "intime": "2180-01-29 18:12:37"}])
ok_out = json.dumps({"returncode": 0, "output": rows})
elided = json.dumps({"returncode": 0, "output_head": rows[:20], "output_tail": rows[20:], "elided_chars": 0, "warning": "Output too long"})
failed = json.dumps({"returncode": 1, "output": "Traceback ..."})


def bash_step(step_id, cmd, content):
    return {"step_id": step_id, "source": "agent",
            "tool_calls": [{"tool_call_id": f"c{step_id}", "function_name": "bash", "arguments": {"command": cmd}}],
            "observation": {"results": [{"source_call_id": f"c{step_id}", "content": content}]}}


helper = "cd /workspace && python3 call_mimic.py list_notes subject_id=12014968 hadm_id=20449658"
readback = "cat /workspace/notes.json"
traj = {"schema_version": "1", "agent": {"name": "mini-swe-agent", "model_name": "m"},
        "steps": [{"step_id": 1, "source": "user", "message": msg_eq},
                  bash_step(2, script, ok_out), bash_step(3, script, failed), bash_step(4, script, elided),
                  bash_step(5, helper, json.dumps({"returncode": 0, "output": "wrote notes.json"})),
                  bash_step(6, readback, json.dumps({"returncode": 0, "output": "{\"note_id\": \"12014968-DS-21\"}"}))]}
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "trajectory.json"; p.write_text(json.dumps(traj))
    t = load_trajectory(p)
tools = [o.tool for o in t.observations]
check("list_icu_stays recovered twice (ok + elided), not from the failed step", tools.count("list_icu_stays") == 2, str(tools))
check("a tool only mentioned in a print string is not a call", "get_lab_timeline" not in tools, str(tools))
check("scope ids carried in args", any(o.tool == "list_icu_stays" and o.args.get("subject_id") == "12014968" for o in t.observations))
check("tool name as a helper-script argument is a call", "list_notes" in tools, str(tools))
pooled = [o for o in t.observations if o.tool == EXEC_OUTPUT_TOOL]
check("stdout pooled once per successful step, incl. steps that name no tool", len(pooled) == 4, str(len(pooled)))
check("a note id read back from a file the agent wrote is in the pool", "12014968-DS-21" in t.evidence_text)
check("elided output pooled as head+tail", any("31929757" in o.text and "18:12:37" in o.text for o in pooled))
check("stay_id from rows is in the evidence pool", "31929757" in t.evidence_text)
st = _scope_stats()
check("R_retrieval credits icustays + notes for the bash trial", {"icustays", "discharge", "radiology"} <= retrieved_tables(t.observations, case_scope(t.steps), st) and st["in_scope"] == 3, str(st))

print()
if _fails:
    print(f"{len(_fails)} FAILED: {_fails}"); sys.exit(1)
print("ALL PASS")
