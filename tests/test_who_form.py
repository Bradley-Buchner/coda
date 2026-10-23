"""Tests for the WHO-form questionnaire strategy (who_form and who_answerer)."""

import re

import pytest

from coda.config import PROMPTS
from coda.questionnaire.bank import load_question_bank
from coda.questionnaire.who_answerer import WhoAnswerer, who_record
from coda.questionnaire.who_form import UNKNOWN, Interview, items_from, load_who_form

ADULT, NEWBORN = 40 * 365.0, 5.0
DIALOGUE = "She had a fever for about a week or two. She died at home."


@pytest.fixture
def form():
    return load_who_form()


def _interview(form, answers, age_days=ADULT, sex="female"):
    """Run an interview to the end. Unlisted yes/no questions are answered no, the rest
    unknown."""
    iv = Interview(form, age_days=age_days, sex=sex)
    while True:
        questions = iv.next_round()
        if not questions:
            return iv
        for q in questions:
            iv.record(q.id, answers.get(q.id, "no" if q.kind == "yesno" else UNKNOWN))


class _FakeLLMClient:
    """Answers by question id; other questions are unknown."""

    def __init__(self, form, answers, fail=False):
        self.text = {q.id: q.text for q in form.questions}
        self.answers, self.fail, self.calls = answers, fail, []

    def call_with_schema(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            return {"api_failed": True}
        by_text = {self.text[qid]: a for qid, a in self.answers.items()}
        questions = re.findall(r"^(q\d+): (.+?) \[", kwargs["user_prompt"], re.M)
        return {key: by_text.get(text, {"evidence": "", "answer": UNKNOWN})
                for key, text in questions}


def test_bank_covers_the_probbase_items(form):
    items = {c["item"] for q in form.questions for c in q.converts}
    expected = {q.id.split(":")[1] for q in load_question_bank()}
    # The age band and day-of-death items (W610022) come from the age
    assert items == {i for i in expected if not i.startswith("W610022")}


def test_sections_follow_age_and_sex(form):
    newborn = _interview(form, {}, age_days=NEWBORN)
    woman = _interview(form, {}, sex="female")
    man = _interview(form, {}, sex="male")
    assert "Id10104" in newborn.answers and "Id10104" not in woman.answers   # did the baby cry
    assert "Id10294" in woman.answers and "Id10294" not in man.answers       # breast lump


def test_follow_ups_open_on_yes(form):
    assert "Id10079" in _interview(form, {"Id10077": "yes"}).answers   # road traffic injury
    assert "Id10079" not in _interview(form, {"Id10077": "no"}).answers
    assert "Id10079" not in _interview(form, {"Id10077": UNKNOWN}).answers


def test_number_answers_convert_to_probbase_items(form):
    items = items_from(form, {"Id10148": {"low": 10, "high": 10, "unit": "days"}})
    assert [items[f"W610148-{s}"][0] for s in "abc"] == ["no", "yes", "no"]


def test_answer_form(form):
    llm = _FakeLLMClient(form, {
        "Id10147": {"evidence": "She had a fever", "answer": "yes"},
        "Id10148": {"evidence": "a fever for about a week or two",
                    "answer": {"low": 7, "high": 14, "unit": "days", "basis": "stated"}},
        "Id10058": {"evidence": "She died in a car", "answer": "Home"},
    })
    answerer = WhoAnswerer(llm, PROMPTS["questionnaire_who_default"])
    answers, interview, complete = answerer.answer_form(DIALOGUE, form, age_days=ADULT,
                                                        sex="female")
    assert complete and len(llm.calls) == len(interview.rounds) > 1
    given = {a.id: a.answer for a in answers}
    assert given["Id10148"]["high"] == 14
    assert given["Id10058"] == UNKNOWN   # its quote isn't in the dialogue
    items = {i["id"]: i["answer"] for i in who_record(answers, interview,
                                                     load_question_bank())["items"]}
    assert items["who.va.q:W610148-a"] == "no"   # 7 to 14 days isn't under a week


def test_failed_call_stops_the_interview(form):
    llm = _FakeLLMClient(form, {}, fail=True)
    answerer = WhoAnswerer(llm, PROMPTS["questionnaire_who_default"])
    answers, _, complete = answerer.answer_form(DIALOGUE, form, age_days=ADULT, sex="female")
    assert not complete and answers == [] and len(llm.calls) == 1
