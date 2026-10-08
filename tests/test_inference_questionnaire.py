"""Tests for questionnaire retrieval inside the CHAMPS prompted agent."""

import asyncio
import re
import threading
import time

import numpy as np

from coda.config import PROMPTS
from coda.inference.champs_prompted_agent import ChampsPromptedInferenceAgent
from coda.questionnaire.answerer import QuestionAnswerer
from coda.questionnaire.bank import Question
from coda.questionnaire.retriever import QuestionRetriever

KEYWORDS = ("fever", "cough", "rash")
QUESTIONS = (
    Question("q:fever-o", "Did he or she have a fever?"),
    Question("q:cough-o", "Did he or she have a cough?"),
    Question("q:rash-o", "Did he or she have a skin rash?"),
)


def _keyword_encoder(texts):
    """Unit vectors counting each keyword, plus a constant so none are zero."""
    vectors = np.array([[t.lower().count(k) for k in KEYWORDS] + [0.1]
                        for t in texts], dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class _FakeLLMClient:
    """Answers questionnaire calls from `answers` (keyed by question text) and
    returns a fixed cause for cause-of-death calls."""

    def __init__(self):
        self.answers = {}
        self.calls = []

    def call_with_schema(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["schema_name"] == "questionnaire_answers":
            questions = re.findall(r"^(q\d+): (.+?) \[", kwargs["user_prompt"], re.M)
            return {key: self.answers.get(text, {"evidence": "", "answer": "unknown"})
                    for key, text in questions}
        return {"reasoning": "r",
                "top_causes": [{"cause_name": "Malaria", "probability": 1.0}],
                "questions": ["q1", "q2", "q3"]}


def _agent(llm, questionnaire=True):
    if not questionnaire:
        return ChampsPromptedInferenceAgent(llm)
    return ChampsPromptedInferenceAgent(
        llm,
        question_retriever=QuestionRetriever(QUESTIONS, encoder=_keyword_encoder,
                                             top_k=3, min_similarity=0.5),
        question_answerer=QuestionAnswerer(llm, PROMPTS["questionnaire_answerer_default"]),
    )


def _cod_prompt(llm):
    return [c for c in llm.calls
            if c["schema_name"] == "champs_cod_classification"][-1]["user_prompt"]


async def test_questionnaire_off_leaves_prompt_unchanged():
    llm = _FakeLLMClient()
    result = await _agent(llm, questionnaire=False).process_chunk(
        "c1", "He had a fever for five days.", [])
    assert len(llm.calls) == 1
    assert "structured VA answers" not in _cod_prompt(llm)
    assert "questionnaire_answers" not in result


async def test_questionnaire_answers_reach_cod_prompt():
    llm = _FakeLLMClient()
    llm.answers = {"Did he or she have a fever?":
                   {"evidence": "He had a fever", "answer": "yes"}}
    result = await _agent(llm).process_chunk("c1", "He had a fever for five days.", [])

    prompt = _cod_prompt(llm)
    assert "structured VA answers" in prompt
    assert "- Did he or she have a fever? yes" in prompt
    assert [(a["id"], a["answer"]) for a in result["questionnaire_answers"]] == \
        [("q:fever-o", "yes")]
    assert set(result["timings"]) == {"questionnaire_retrieval_s",
                                      "questionnaire_answering_s", "inference_s"}


async def test_answers_accumulate_across_chunks():
    llm = _FakeLLMClient()
    agent = _agent(llm)
    llm.answers = {"Did he or she have a fever?":
                   {"evidence": "He had a fever", "answer": "yes"}}
    await agent.process_chunk("c1", "He had a fever for five days.", [])

    # A later "unknown" keeps the earlier answer; new answers are added
    llm.answers = {"Did he or she have a cough?":
                   {"evidence": "a bad cough", "answer": "yes"}}
    await agent.process_chunk("c2", "Then a bad cough started.", [])

    # A later yes/no replaces the earlier answer
    llm.answers = {"Did he or she have a fever?":
                   {"evidence": "no fever", "answer": "no"}}
    result = await agent.process_chunk("c3", "Sorry, he had no fever.", [])

    assert [(a["id"], a["answer"]) for a in result["questionnaire_answers"]] == \
        [("q:fever-o", "no"), ("q:cough-o", "yes")]


async def test_questionnaire_failure_still_returns_causes():
    llm = _FakeLLMClient()
    agent = _agent(llm)

    def fake_retrieve(text):
        raise RuntimeError("boom")

    agent.question_retriever.retrieve = fake_retrieve
    result = await agent.process_chunk("c1", "He had a fever.", [])
    assert result["causes"]
    assert result["questionnaire_answers"] == []


async def test_session_agents_share_retriever_but_not_answers():
    llm = _FakeLLMClient()
    llm.answers = {"Did he or she have a fever?":
                   {"evidence": "He had a fever", "answer": "yes"}}
    agent = _agent(llm)
    session = agent.create_session_agent()
    await session.process_chunk("c1", "He had a fever.", [])

    assert session.question_retriever is agent.question_retriever
    assert session.question_answerer is agent.question_answerer
    assert list(session.questionnaire_answers) == ["q:fever-o"]
    assert agent.questionnaire_answers == {}
    session.reset()
    assert session.questionnaire_answers == {}


async def test_questionnaire_work_runs_off_event_loop_within_llm_bound():
    counter = {"active": 0, "max_active": 0, "lock": threading.Lock()}
    threads = []

    class SlowLLMClient(_FakeLLMClient):
        def call_with_schema(self, **kwargs):
            threads.append(threading.get_ident())
            with counter["lock"]:
                counter["active"] += 1
                counter["max_active"] = max(counter["max_active"], counter["active"])
            try:
                time.sleep(0.1)
                return super().call_with_schema(**kwargs)
            finally:
                with counter["lock"]:
                    counter["active"] -= 1

    llm = SlowLLMClient()
    llm.answers = {"Did he or she have a fever?":
                   {"evidence": "He had a fever", "answer": "yes"}}
    prototype = _agent(llm)
    retrieve = prototype.question_retriever.retrieve

    def fake_retrieve(text):
        threads.append(threading.get_ident())
        return retrieve(text)

    prototype.question_retriever.retrieve = fake_retrieve
    await asyncio.gather(
        prototype.create_session_agent().process_chunk("a", "He had a fever.", []),
        prototype.create_session_agent().process_chunk("b", "He had a fever.", []),
    )

    # Per session: retrieval, the questionnaire LLM call and the COD LLM call
    assert len(threads) == 6
    assert threading.get_ident() not in threads
    assert counter["max_active"] == 1
