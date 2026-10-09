"""Tests for questionnaire retrieval (coda.questionnaire)."""

import numpy as np
import pandas as pd
import pytest

from coda.config import PROMPTS
from coda.questionnaire.answerer import (
    QuestionAnswer,
    QuestionAnswerer,
    build_answer_schema,
    format_answers,
)
from coda.questionnaire.bank import (
    UNKNOWN,
    Question,
    load_question_bank,
    question_bank_from_probbase,
)
from coda.questionnaire.retriever import QuestionRetriever, split_into_units

KEYWORDS = ("fever", "cough", "rash")
QUESTIONS = (
    Question("q:fever", "Did he or she have a fever?"),
    Question("q:cough", "Did he or she have a cough?"),
    Question("q:rash", "Did he or she have a skin rash?"),
)


def _keyword_encoder(texts):
    """Unit vectors counting each keyword, plus a constant so none are zero."""
    vectors = np.array([[t.lower().count(k) for k in KEYWORDS] + [0.1]
                        for t in texts], dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _write_bank(path, rows, columns=("id", "question", "responses", "category")):
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, sep="\t", index=False)
    return path


def test_default_bank_loads():
    questions = load_question_bank()
    assert len(questions) == 341
    # Regression: maternal and newborn delivery questions shared identical text
    assert len({q.text for q in questions}) == len(questions)
    # Regression: the embedder reads "(s)he" as "he"
    assert not any("(s)he" in q.text or "his/her" in q.text for q in questions)


def test_custom_bank_responses(tmp_path):
    bank = _write_bank(tmp_path / "bank.tsv", [
        ["q:1", "Did she have a fever?", "", "symptom"],
        ["q:2", "Where did she die?", "home|hospital|other", ""],
    ])
    fever, place = load_question_bank(bank)
    assert fever.allowed_answers == ("yes", "no", UNKNOWN)
    assert place.allowed_answers == ("home", "hospital", "other", UNKNOWN)


def test_malformed_bank_rejected(tmp_path):
    bank = _write_bank(tmp_path / "bank.tsv", [["q:1", "Fever?"], ["q:1", "Cough?"]],
                       ("id", "question"))
    with pytest.raises(ValueError):
        load_question_bank(bank)


def test_bank_from_probbase_keeps_symptoms_and_background(tmp_path):
    nodes = tmp_path / "probbase_nodes.tsv"
    pd.DataFrame({
        "id:ID": ["who.va.q:W610147-o", "who.va.q:W610411-o",
                  "who.va.q:W610019-a", "who.va.q:W610344-o",
                  "who.va.q:W610252-o"],
        "name": ["Did (s)he have a fever?", "Did (s)he drink alcohol?",
                 "Was he male?", "Was the delivery a Caesarean section?",
                 "Did s/he have puffiness all over his/her body?"],
        "samb": ["M", "B", "S", "M", "M"],
    }).to_csv(nodes, sep="\t", index=False)

    question_bank_from_probbase(nodes, tmp_path / "bank.tsv")

    questions = {q.id: q for q in load_question_bank(tmp_path / "bank.tsv")}
    assert set(questions) == {"who.va.q:W610147-o", "who.va.q:W610411-o",
                              "who.va.q:W610344-o", "who.va.q:W610252-o"}
    assert questions["who.va.q:W610147-o"].text == "Did he or she have a fever?"
    # Regression: replacing "s/he" first turned "his/her" into "hihe or sher"
    assert questions["who.va.q:W610252-o"].text == \
        "Did he or she have puffiness all over his or her body?"
    # Regression: maternal and newborn Caesarean questions shared identical text
    assert questions["who.va.q:W610344-o"].text == \
        "Did she give birth by Caesarean section?"


def test_split_into_units_overlapping_windows():
    text = "He had a fever. Then a cough started.\nHe died at home."
    assert split_into_units(text, window=2) == [
        "He had a fever. Then a cough started.",
        "Then a cough started. He died at home.",
    ]


def test_split_into_units_cuts_long_sentences():
    units = split_into_units("word " * 100, window=1, max_words=40)
    assert [len(u.split()) for u in units] == [40, 40, 20]


def test_retrieve_ranks_matching_questions():
    retriever = QuestionRetriever(QUESTIONS, encoder=_keyword_encoder,
                                  top_k=2, min_similarity=0.5)
    retrieved = retriever.retrieve("She had a high fever. The fever lasted a week.")
    assert [r.question.id for r in retrieved] == ["q:fever"]


def test_retrieve_empty_text():
    retriever = QuestionRetriever(QUESTIONS, encoder=_keyword_encoder)
    assert retriever.retrieve("") == []


def test_retrieve_adds_other_options_of_retrieved_item():
    questions = [
        Question("q:cough-o", "Did he or she have a cough?"),
        Question("q:fever-a", "Did the fever last less than a week?"),
        Question("q:fever-b", "Did the fever last one to two weeks?"),
        Question("q:fever-c", "Did the fever last more than two weeks?"),
    ]
    retriever = QuestionRetriever(questions, encoder=_keyword_encoder,
                                  top_k=1, min_similarity=0.5)
    retrieved = retriever.retrieve("He had a fever for ten days.")
    assert [r.question.id for r in retrieved] == ["q:fever-a", "q:fever-b", "q:fever-c"]
    assert [r.sibling_of for r in retrieved] == [None, "q:fever-a", "q:fever-a"]


DIALOGUE = "Respondent: He had a high fever for five days and a bad cough."
FEVER = Question("q:fever-o", "Did he or she have a fever?")
FEVER_2W = Question("q:fev2w-o", "Did the fever last at least 2 weeks?")
PLACE = Question("q:place-o", "Where did he or she die?", responses=("home", "hospital"))


class _FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call_with_schema(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _answer(response, questions=(FEVER, FEVER_2W), dialogue=DIALOGUE):
    llm = _FakeLLMClient(response)
    answerer = QuestionAnswerer(llm, PROMPTS["questionnaire_answerer_default"])
    answers = answerer.answer(dialogue, list(questions))
    return answers, llm


def test_answer_schema_restricts_each_question():
    schema = build_answer_schema([FEVER, PLACE])
    assert schema["properties"]["q1"]["properties"]["answer"]["enum"] == \
        ["yes", "no", UNKNOWN]
    assert schema["properties"]["q2"]["properties"]["answer"]["enum"] == \
        ["home", "hospital", UNKNOWN]


def test_answer_keeps_supported_answers():
    answers, llm = _answer({
        "q1": {"evidence": "He had a high fever", "answer": "Yes."},
        "q2": {"evidence": "a high fever for five days", "answer": "no"},
    })
    assert [(a.id, a.answer) for a in answers] == [("q:fever-o", "yes"),
                                                  ("q:fev2w-o", "no")]
    prompt = llm.calls[0]["user_prompt"]
    assert DIALOGUE in prompt
    assert "q2: Did the fever last at least 2 weeks? [yes | no | unknown]" in prompt


def test_answer_downgrades_unsupported_answers():
    answers, _ = _answer({
        "q1": {"evidence": "She was burning hot", "answer": "yes"},  # not in dialogue
        "q2": {"evidence": "", "answer": "maybe"},                    # not allowed
    })
    assert [(a.answer, a.evidence) for a in answers] == [(UNKNOWN, ""), (UNKNOWN, "")]


def _evidence_kept(dialogue, evidence):
    answers, _ = _answer({"q1": {"evidence": evidence, "answer": "yes"}},
                         questions=[FEVER], dialogue=dialogue)
    return answers[0].answer == "yes"


def test_answer_requires_exact_evidence():
    assert _evidence_kept("He had a HIGH fever, for days.", "he had a high fever for days")
    # Regression: fuzzy matching accepted these (negation dropped, partial word)
    assert not _evidence_kept("Respondent: He was not vomiting.", "he was vomiting")
    assert not _evidence_kept("He died in a car crash.", "rash")
    # Regression: straight and curly apostrophes broke correct quotes
    assert _evidence_kept("She told me 'he had a fever' every night.", "he had a fever")
    assert _evidence_kept("He wasn’t coughing.", "he wasn't coughing")


def test_answer_returns_nothing_when_llm_fails():
    answers, _ = _answer({"api_failed": True})
    assert answers == []


def test_format_answers_one_line_per_statement():
    quote = "He had a high fever for five days"
    answers = [
        QuestionAnswer("q:147-o", "Did he or she have a fever?", "yes", quote),
        QuestionAnswer("q:148-a", "Did the fever last less than a week?", "yes", quote),
        QuestionAnswer("q:148-b", "Did the fever last 1 to 2 weeks?", "no", quote),
        QuestionAnswer("q:150-a", "Was the fever severe?", "yes", "a high fever"),
        QuestionAnswer("q:182-a", "Diarrhoea for less than 2 weeks?", "no", "no diarrhoea"),
        QuestionAnswer("q:182-b", "Diarrhoea for at least 2 weeks?", "no", "no diarrhoea"),
        QuestionAnswer("q:153-o", "Did he or she have a cough?", UNKNOWN, ""),
    ]
    # Implied "no" options are dropped, and overlapping quotes share one line
    assert format_answers(answers) == (
        f'- "{quote}": Did he or she have a fever? yes; '
        "Did the fever last less than a week? yes; Was the fever severe? yes")

