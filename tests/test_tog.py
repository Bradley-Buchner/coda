import pytest

from coda.inference.champs_prompted.base_agent import LLMCallFailed
from coda.inference.tog import ToGBackwardIcd10Agent
from coda.inference.tog.backward import (
    MAX_HOPS,
    check_findings,
    format_evidence,
    walk_back,
)


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
    call = FakeCall([{"findings": [{"number": 2, "status": "yes"}]}])
    findings = await check_findings(FakeGraph(), call, "Fever.", "Sepsis",
                                    "A41.9")
    assert [(f["finding"], f["status"]) for f in findings] == \
        [("cough", "not_mentioned"), ("fever", "yes")]
    assert "W610022" not in call.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_failed_finding_check_is_not_evidence():
    call = FakeCall([None])
    findings = await check_findings(FakeGraph(), call, "", "Sepsis", "A41.9")
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
            "finding_check": [{"findings": [{"number": 2,
                                             "status": "yes"}]}],
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
    assert result["reasoning"] == "Fever fits sepsis."
    evidence = result["kg_evidence"]
    assert evidence["cited"] == ["fever"]
    assert [c["cause"] for c in evidence["candidates"]] == \
        ["Sepsis", "Lower respiratory infections"]
    assert [t["step"] for t in agent.trace] == [
        "first_opinion", "walk_step", "finding_check", "walk_step",
        "final_ranking"]
    assert "[yes] fever (grade A+ for sepsis on answer yes)" in \
        client.calls[-1]["user_prompt"]
    assert "Same as: ICD-11 1G40 Sepsis" in client.calls[-1]["user_prompt"]
    assert result["questions"] == ["q1", "q2", "q3"]
    assert result["chunk_id"] == "c1" and "timings" in result


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
