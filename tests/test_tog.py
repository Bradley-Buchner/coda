import pandas as pd
import pytest

from coda.inference.champs_prompted.base_agent import LLMCallFailed
from coda.inference.tog import (
    ToGBackwardIcd10Agent,
    ToGFindingsIcd10Agent,
    ToGReasoningIcd10Agent,
)
from coda.inference.tog.agent import REASONING_SCHEMA
from coda.inference.tog.backward import (
    MAX_HOPS,
    check_findings,
    checked_status,
    format_evidence,
    normalized,
    walk_back,
)
from coda.inference.tog.reasoning import (
    NO_EVIDENCE,
    check_step,
    diff_steps,
    kg_items,
    render_items,
    stable_changes,
)
from coda.kg.check import check_top_causes
from coda.kg.query import KGQuery
from coda.kg.sources import write_tsv_gz


class FakeGraph:
    """A chain A41.9 -> A41 -> A40-A49 -> A00-B99 plus a causes edge."""

    parents = {"icd10:A41.9": "icd10:A41", "icd10:A41": "icd10:A40-A49",
               "icd10:A40-A49": "icd10:A00-B99", "icd10:J18.9": "icd10:J18"}
    who_va = {"A41.9": ["who.va:sepsis"]}
    findings = {"who.va:sepsis": [("who.va.q:fever", "A+"),
                                  ("who.va.q:cough", "A-"),
                                  ("who.va.q:W610022-e", "A")]}

    def same_as(self, code):
        return "ICD-11 1G40 Sepsis" if code == "A41.9" else ""

    def resolve(self, code):
        node = f"icd10:{code}"
        return node if node in self.parents else None

    def name(self, node):
        return node.split(":", 1)[-1]

    def relations(self, node):
        rels = [("is_a", "out")] if node in self.parents else []
        if node == "icd10:A41.9":
            rels.append(("causes", "out"))
        return rels

    def neighbors(self, node, relation, direction):
        if relation == "is_a":
            return [self.parents[node]]
        return ["icd10:R57.2"]

    def who_va_causes(self, code):
        return self.who_va.get(code, [])

    def strong_findings(self, cause):
        return self.findings[cause]

    def graded_answer(self, question):
        return "yes"


class FakeCall:
    """Returns scripted responses and records each call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, step, system_prompt, user_prompt, schema,
                       **context):
        self.calls.append({"step": step, "user_prompt": user_prompt,
                           **context})
        response = self.responses.pop(0)
        if response is None:
            raise LLMCallFailed("failed")
        return response


async def walk(responses):
    call = FakeCall(responses)
    triples = await walk_back(FakeGraph(), call, "## INPUT\nFever.",
                              "Sepsis", "A41.9")
    return triples, call


@pytest.mark.asyncio
async def test_walk_follows_chosen_facts_and_stops_on_none():
    triples, call = await walk([{"chosen": [1], "reason": ""},
                                {"chosen": [], "reason": ""}])
    assert triples == ["(A41.9, is_a, A41)"]
    assert [c["hop"] for c in call.calls] == [1, 2]
    assert call.calls[0]["options"] == ["(A41.9, is_a, A41)",
                                        "(A41.9, causes, R57.2)"]
    assert call.calls[0]["user_prompt"].startswith("## INPUT\nFever.")


@pytest.mark.asyncio
async def test_walk_stops_after_max_hops_and_skips_bad_numbers():
    triples, call = await walk([{"chosen": [1, 99], "reason": ""}] * 5)
    assert len(call.calls) == MAX_HOPS
    assert triples[-1] == "(A40-A49, is_a, A00-B99)"


@pytest.mark.asyncio
async def test_finding_check_maps_statuses_and_skips_age_brackets():
    call = FakeCall([{"findings": [
        {"number": 1, "status": "no", "quote": "no cough"},
        {"number": 2, "status": "yes", "quote": " FEVER. "}]}])
    [findings] = await check_findings(FakeGraph(), call, "Fever.", ["A41.9"])
    assert [(f["finding"], f["status"]) for f in findings] == \
        [("cough", "not_mentioned"), ("fever", "yes")]
    assert "W610022" not in call.calls[0]["user_prompt"]
    case_text = normalized("Fever, and vomited twice. Then she died.")
    assert checked_status({"status": "yes", "quote": "she vomited twice ... "
                           "she died"}, case_text) == "yes"
    assert checked_status({"status": "no", "quote": "she vomited ... no rash"},
                          case_text) == "not_mentioned"


@pytest.mark.asyncio
async def test_candidates_share_one_finding_check():
    class TwoCauses(FakeGraph):
        who_va = {"A41.9": ["who.va:sepsis"], "J18.9": ["who.va:lri"]}
        findings = {"who.va:sepsis": [("who.va.q:fever", "A+")],
                    "who.va:lri": [("who.va.q:cough", "A"),
                                   ("who.va.q:fever", "A-"),
                                   ("who.va.q:W610310-o", "A+")]}

    call = FakeCall([{"findings": [{"number": 2, "status": "yes",
                                    "quote": "Fever"}]}])
    sepsis, lri = await check_findings(TwoCauses(), call, "Fever.",
                                       ["A41.9", "J18.9"])
    assert len(call.calls) == 1
    assert "W610310" not in call.calls[0]["user_prompt"]
    assert [(f["finding"], f["grade"], f["status"]) for f in sepsis] == \
        [("fever", "A+", "yes")]
    assert [(f["finding"], f["grade"], f["status"]) for f in lri] == \
        [("cough", "A", "not_mentioned"), ("fever", "A-", "yes")]


@pytest.mark.asyncio
async def test_failed_finding_check_is_not_evidence():
    call = FakeCall([None])
    [findings] = await check_findings(FakeGraph(), call, "", ["A41.9"])
    assert findings is None
    text = format_evidence({"cause": "Sepsis", "code": "A41.9",
                            "score": 0.5, "same_as": "", "triples": [],
                            "findings": findings})
    assert "(finding check failed)" in text


FIRST_OPINION = {"reasoning": "Fever.", "questions": ["q1", "q2", "q3"],
                 "top_causes": [
                     {"cause_name": "Sepsis", "probability": 0.7},
                     {"cause_name": "Lower respiratory infections",
                      "probability": 0.3}]}
FINAL_RANKING = {"reasoning": "Fever fits sepsis.",
                 "candidate_updates": [
                     {"cause_name": "Sepsis", "direction": "up",
                      "evidence": "Fever was present."}],
                 "cited_facts": [],
                 "cited_findings": ["[yes] fever (grade A+ for sepsis)",
                                    "hot to the touch"],
                 "top_causes": [
                     {"cause_name": "Sepsis", "probability": 0.9},
                     {"cause_name": "Lower respiratory infections",
                      "probability": 0.1}]}


class FakeLLMClient:
    """Answers each schema_name with the next scripted response."""

    def __init__(self, final_ranking=FINAL_RANKING):
        self.responses = {
            "champs_cod_classification": [FIRST_OPINION],
            "walk_step": [{"chosen": [], "reason": ""}] * 2,
            "finding_check": [{"findings": [{"number": 2, "status": "yes",
                                             "quote": "Fever"}]}],
            "final_ranking": [final_ranking],
        }
        self.calls = []

    def call_with_schema(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses[kwargs["schema_name"]].pop(0)


async def run(final_ranking=FINAL_RANKING):
    client = FakeLLMClient(final_ranking)
    agent = ToGBackwardIcd10Agent(client, graph=FakeGraph())
    result = await agent.process_chunk("c1", "Fever.", [])
    return result, agent, client


@pytest.mark.asyncio
async def test_ranks_candidates_with_kg_evidence():
    result, agent, client = await run()

    assert list(result["causes"]) == ["icd10:A41.9", "icd10:J18.9"]
    assert result["causes"]["icd10:A41.9"]["score"] == pytest.approx(0.9)
    assert result["reasoning"] == (
        "Fever fits sepsis. Sepsis moved from 0.70 to 0.90: Fever was present. "
        "Lower respiratory infections moved from 0.30 to 0.10: no reason given.")
    evidence = result["kg_evidence"]
    assert evidence["cited"] == ["fever"]
    assert [c["cause"] for c in evidence["candidates"]] == \
        ["Sepsis", "Lower respiratory infections"]
    assert [t["step"] for t in agent.trace] == [
        "first_opinion", "walk_step", "walk_step", "finding_check",
        "final_ranking"]
    assert "[yes] fever (grade A+ for sepsis on answer yes)" in \
        client.calls[-1]["user_prompt"]
    assert "cough" not in client.calls[-1]["user_prompt"]
    assert "(1 other typical findings were answered otherwise" in \
        client.calls[-1]["user_prompt"]
    assert "Same as: ICD-11 1G40 Sepsis" in client.calls[-1]["user_prompt"]
    assert result["questions"] == ["q1", "q2", "q3"]
    assert result["chunk_id"] == "c1" and "timings" in result


@pytest.mark.asyncio
async def test_no_change_keeps_the_first_opinion_probability():
    result, _, _ = await run({**FINAL_RANKING, "candidate_updates": [
        {"cause_name": "Sepsis", "direction": "no_change", "evidence": "None."},
        {"cause_name": "Lower respiratory infections", "direction": "no_change",
         "evidence": "None."}]})
    assert result["causes"]["icd10:A41.9"]["score"] == pytest.approx(0.7)
    assert result["reasoning"] == "Fever fits sepsis."


@pytest.mark.asyncio
async def test_flags_a_direction_the_numbers_contradict():
    result, _, _ = await run({**FINAL_RANKING, "candidate_updates": [
        {"cause_name": "Sepsis", "direction": "down", "evidence": "Why."}]})
    assert "Sepsis moved from 0.70 to 0.90: Why. (Stated direction: down.)" \
        in result["reasoning"]


@pytest.mark.asyncio
async def test_keeps_cited_facts_and_findings_from_separate_lists():
    client = FakeLLMClient({**FINAL_RANKING,
                            "cited_facts": ["(A41.9, is_a, A41)"]})
    client.responses["walk_step"] = [{"chosen": [1], "reason": ""},
                                     {"chosen": [], "reason": ""},
                                     {"chosen": [], "reason": ""}]
    agent = ToGBackwardIcd10Agent(client, graph=FakeGraph())
    result = await agent.process_chunk("c1", "Fever.", [])
    assert result["kg_evidence"]["cited"] == ["(A41.9, is_a, A41)", "fever"]


@pytest.mark.asyncio
@pytest.mark.parametrize("final_ranking", [
    {"api_failed": True},
    {**FINAL_RANKING, "top_causes": [
        {"cause_name": "Pneumonia of the left lung", "probability": 1}]},
])
async def test_keeps_first_opinion_if_final_ranking_fails(final_ranking):
    result, _, _ = await run(final_ranking)
    assert result["causes"]["icd10:A41.9"]["score"] == pytest.approx(0.7)
    assert result["reasoning"] == "Fever."
    assert result["kg_evidence"]["cited"] == []


@pytest.mark.asyncio
async def test_findings_agent_ranks_with_findings_alone():
    client = FakeLLMClient()
    agent = ToGFindingsIcd10Agent(client, graph=FakeGraph())
    result = await agent.process_chunk("c1", "Fever.", [])
    assert [t["step"] for t in agent.trace] == [
        "first_opinion", "finding_check", "final_ranking"]
    final = client.calls[-1]
    assert "Walk facts" not in final["user_prompt"]
    assert "Same as" not in final["user_prompt"]
    assert "cited_facts" not in final["schema"]["properties"]
    assert result["kg_evidence"]["cited"] == ["fever"]


EDGES = [
    # :START_ID, :END_ID, :TYPE, maybe, rule_type, result, reason
    ["icd10:A41.9", "icd10:J18.9", "causes", "False", "", "", ""],
    ["icd10:A41.9", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:J18.9", "acme:table_A", "in_code_list", "", "", "", ""],
]
SEPSIS, LRI = "Sepsis", "Lower respiratory infections"
CAUSES = {"icd10:A41.9": {"name": SEPSIS}, "icd10:J18.9": {"name": LRI}}


@pytest.fixture
def kg(tmp_path):
    columns = [":START_ID", ":END_ID", ":TYPE", "maybe:boolean", "rule_type",
               "result", "reason"]
    write_tsv_gz(pd.DataFrame(EDGES, columns=columns),
                 tmp_path / "acme_edges.tsv.gz")
    write_tsv_gz(pd.DataFrame([], columns=["id:ID", "name"]),
                 tmp_path / "icd10_nodes.tsv.gz")
    return KGQuery(kg_dir=tmp_path)


def finding(name, grade, status, who_va="sepsis"):
    return {"finding": name, "grade": grade, "graded_answer": "yes",
            "who_va_cause": who_va, "status": status}


def test_kg_items_number_rules_findings_and_walks(kg):
    candidates = [
        {"cause": SEPSIS, "triples": ["(A41.9, is_a, A41)"],
         "findings": [finding("fever", "A+", "yes")]},
        {"cause": LRI, "triples": ["(A41.9, is_a, A41)", "(J18.9, is_a, J18)"],
         "findings": [finding("fever", "A-", "yes", "pneumonia"),
                      finding("cough", "A", "not_mentioned", "pneumonia")]},
    ]
    items = kg_items(check_top_causes(kg, CAUSES), CAUSES, candidates,
                     walks=True)
    assert [(i["id"], i["text"]) for i in items] == [
        ("R1", f"{LRI} may be due to {SEPSIS} (ACME Table D). If both are "
               f"reported, {SEPSIS} is the underlying cause."),
        ("F1", f"[yes] fever Typical of {SEPSIS} (grade A+ for sepsis on "
               f"answer yes); {LRI} (grade A- for pneumonia on answer yes)"),
        ("W1", "(A41.9, is_a, A41)"),
        ("W2", "(J18.9, is_a, J18)")]
    assert (items[0]["cause"], items[0]["consequence"]) == (SEPSIS, LRI)
    assert items[1]["typical"] == [SEPSIS, LRI]
    assert items[2]["candidate"] == SEPSIS
    assert [i["id"] for i in kg_items(check_top_causes(kg, CAUSES), CAUSES,
                                      candidates)] == ["R1", "F1"]
    assert NO_EVIDENCE in render_items([])
    injury = {"links": [], "flags": [
        {"curie": "icd10:T14.9", "code": "icd10:T14.9", "table": "B",
         "flag": "can only be a contributing cause, not the underlying cause"}]}
    assert kg_items(injury, {"icd10:T14.9": {"name": "Injury"}}, []) == []


ITEMS = [
    {"id": "R1", "kind": "R", "cause": SEPSIS, "consequence": LRI,
     "both_ways": False, "text": f"{LRI} may be due to {SEPSIS}."},
    {"id": "R2", "kind": "R", "cause": SEPSIS, "consequence": LRI,
     "both_ways": True},
    {"id": "R3", "kind": "R", "candidate": "Injury", "table": "B"},
    {"id": "F1", "kind": "F", "typical": [SEPSIS]},
    {"id": "W1", "kind": "W", "candidate": SEPSIS},
    {"id": "F2", "kind": "F", "typical": [SEPSIS, LRI]},
    {"id": "R4", "kind": "R", "candidate": "Upper respiratory infections",
     "table": "H"},
]


def step(basis, favors=(), disfavors=(), underlying="", role="other",
         check="consistent", valid=()):
    return {"claim": "", "role": role, "favors": list(favors),
            "disfavors": list(disfavors), "underlying": underlying,
            "basis": list(basis), "quote": "", "check": check,
            "valid": list(valid or [i for i in basis if i != "narrative"])}


@pytest.mark.parametrize("cited, expected", [
    (step(["R1"], [SEPSIS], [LRI], SEPSIS), "consistent"),
    (step(["R1"], [LRI, SEPSIS], [], LRI), "inconsistent"),
    # The rule's cause is argued away, so the rule doesn't apply
    (step(["R1"], [LRI], [SEPSIS], LRI), "unchecked"),
    (step(["R1"], [SEPSIS]), "unchecked"),
    (step(["R1"], [LRI], [], LRI, role="not_applicable"), "rejected"),
    (step(["R2"], [LRI], [SEPSIS], LRI), "consistent"),
    (step(["R3"], ["Injury"]), "inconsistent"),
    (step(["R3"], [SEPSIS], ["Injury"], SEPSIS), "consistent"),
    (step(["F1"], [SEPSIS]), "consistent"),
    (step(["F1"], [LRI], [SEPSIS]), "inconsistent"),
    # A finding typical of both doesn't tell them apart
    (step(["F2"], [SEPSIS], [LRI]), "inconsistent"),
    (step(["F2"], [], [], role="tells_apart"), "unchecked"),
    (step(["F1"], [], [SEPSIS], role="not_applicable"), "rejected"),
    # A trivial condition is ruled out only by another condition in play
    (step(["R4"], ["Upper respiratory infections"]), "consistent"),
    (step(["R4"], ["Upper respiratory infections"], [], LRI), "inconsistent"),
    (step(["F1", "R1"], [], [SEPSIS], LRI), "inconsistent"),
    (step(["W1"], [SEPSIS]), "consistent"),
    (step(["R9", "narrative"], [SEPSIS]), "unchecked"),
])
def test_check_step(cited, expected):
    assert check_step(cited, ITEMS) == expected


def test_diff_and_stable_changes():
    sequence = step(["R1"], [SEPSIS], [LRI], SEPSIS, role="sequence")
    fits = step(["narrative"], [SEPSIS], role="finding_fit",
                check="unchecked")
    # Both present, yet the rule's consequence is taken as underlying
    wrong = step(["narrative"], [LRI, SEPSIS], [], LRI, role="sequence",
                 check="unchecked")
    injury = step(["narrative"], ["Injury"], role="other", check="unchecked")
    diff = diff_steps([sequence, {**fits, "basis": ["F1"], "valid": ["F1"],
                                  "check": "consistent"}], [wrong, fits, injury])
    assert diff["added"] == []
    assert diff["changed"] == [("sequence", frozenset({SEPSIS, LRI}))]
    assert diff["omitted"] == [("other", frozenset({"Injury"}))]
    assert diff["grounded"] == [("finding_fit", frozenset({SEPSIS}))]
    changes = stable_changes([[sequence], [sequence]], [[wrong], [wrong]],
                             ITEMS)
    assert changes["provable_changed"] == \
        [("sequence", frozenset({SEPSIS, LRI}))]
    assert changes["kinds"][("sequence", frozenset({SEPSIS, LRI}))] == ["R"]
    # An omitted step must be in every control run, and is provable if a
    # shown rule contradicts it
    changes = stable_changes([[], []], [[wrong], []], ITEMS)
    assert changes["omitted"] == []
    changes = stable_changes([[], []], [[wrong], [wrong]], ITEMS)
    assert changes["provable_omitted"] == \
        [("sequence", frozenset({SEPSIS, LRI}))]
    changes = stable_changes([[], []], [[sequence], [sequence]], ITEMS)
    assert changes["omitted"] and not changes["provable_omitted"]
    # An added step must be in every KG run
    changes = stable_changes([[sequence], []], [[], []], ITEMS)
    assert changes["added"] == []


REASONING = {
    "reasoning_steps": [
        {"claim": "Sepsis led to the pneumonia.", "role": "sequence",
         "favors": [SEPSIS], "disfavors": ["Lower respiratory infection"],
         "underlying": SEPSIS, "basis": ["R1", "narrative"],
         "quote": "fever"},
        {"claim": "Fever fits sepsis.", "role": "finding_fit",
         "favors": [SEPSIS], "disfavors": [], "underlying": "",
         "basis": ["F1", "R7"], "quote": ""}],
    "candidate_updates": [{"cause_name": SEPSIS, "direction": "no_change",
                           "evidence": "None."},
                          {"cause_name": LRI, "direction": "no_change",
                           "evidence": "None."}],
    "top_causes": FINAL_RANKING["top_causes"]}


@pytest.mark.asyncio
async def test_reasoning_agent_checks_steps_against_kg_items(kg):
    client = FakeLLMClient(REASONING)
    agent = ToGReasoningIcd10Agent(client, graph=FakeGraph(), kg=kg)
    result = await agent.process_chunk("c1", "Fever.", [])
    assert [t["step"] for t in agent.trace] == [
        "first_opinion", "finding_check", "final_ranking"]
    final = client.calls[-1]
    assert final["schema"] == REASONING_SCHEMA
    assert "R1. Lower respiratory infections may be due to Sepsis" in \
        final["user_prompt"]
    assert "F1. [yes] fever Typical of Sepsis" in final["user_prompt"]
    assert "Walk facts" not in final["user_prompt"]
    evidence = result["kg_evidence"]
    first, second = evidence["steps"]
    assert first["disfavors"] == [LRI]
    assert (first["valid"], first["check"], first["quote_ok"]) == \
        (["R1"], "consistent", True)
    assert (second["valid"], second["invalid"], second["check"],
            second["quote_ok"]) == (["F1"], ["R7"], "consistent", None)
    assert len(evidence["cited"]) == 2
    assert result["reasoning"] == "Sepsis led to the pneumonia. Fever fits " \
                                  "sepsis."
    assert result["causes"]["icd10:A41.9"]["score"] == pytest.approx(0.7)
    assert agent.create_session_agent().kg is kg


@pytest.mark.asyncio
async def test_reasoning_arms_differ_only_in_the_kg_items(kg):
    client = FakeLLMClient()
    client.responses["final_ranking"] = [REASONING, REASONING]
    agent = ToGReasoningIcd10Agent(client, graph=FakeGraph(), kg=kg)
    first_opinion = {"causes": {
        "icd10:A41.9": {"name": SEPSIS, "score": 0.7},
        "icd10:J18.9": {"name": LRI, "score": 0.3}}, "reasoning": ""}
    candidates = [{"cause": SEPSIS, "code": "A41.9", "score": 0.7,
                   "findings": []},
                  {"cause": LRI, "code": "J18.9", "score": 0.3,
                   "findings": []}]
    for items in ([], [ITEMS[0]]):
        await agent.rank("c1", first_opinion, "## INPUT\nFever.", candidates,
                         items=items)
    control, shown = client.calls
    assert control["system_prompt"] == shown["system_prompt"]
    assert control["schema"] == shown["schema"]
    assert NO_EVIDENCE in control["user_prompt"]
    assert control["user_prompt"].split("## KG EVIDENCE")[0] == \
        shown["user_prompt"].split("## KG EVIDENCE")[0]
