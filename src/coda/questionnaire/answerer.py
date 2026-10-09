"""Answer the retrieved questionnaire questions from the dialogue with an LLM.

The LLM is asked for structured output: one entry per question, each with a
verbatim evidence quote followed by an answer restricted to that question's
allowed answers.

A yes/no (or other non-"unknown") answer is kept only if its surfaced evidence
quote appears word for word in the dialogue (ignoring case and punctuation);
otherwise it is downgraded to "unknown".

Quotes must be exact: a fuzzy match would accept "he was vomiting" for "he was
not vomiting".
"""
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from coda.llm_api.client import LLMClient
from coda.questionnaire.bank import OPTION_ID, Question, UNKNOWN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionAnswer:
    id: str
    question: str
    answer: str
    evidence: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def build_answer_schema(questions: Sequence[Question]) -> Dict[str, Any]:
    """Build the JSON schema: an {evidence, answer} entry per question (q1, q2, ...)."""
    properties = {
        f"q{i}": {
            "type": "object",
            "properties": {
                "evidence": {"type": "string"},
                "answer": {"type": "string", "enum": list(q.allowed_answers)},
            },
            "required": ["evidence", "answer"],
            "additionalProperties": False,
        }
        for i, q in enumerate(questions, 1)
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def format_questions(questions: Sequence[Question]) -> str:
    """Format the questions as numbered prompt lines with their allowed answers."""
    return "\n".join(f"q{i}: {q.text} [{' | '.join(q.allowed_answers)}]"
                     for i, q in enumerate(questions, 1))


def format_answers(answers: Iterable[QuestionAnswer]) -> str:
    """Format known answers as prompt lines, one line per quoted statement.

    Answers whose quotes overlap come from one statement, so they share a line.
    When several options of one question are answered, only the "yes" options
    are kept, since the "no" ones are implied. "unknown" answers are left out.
    """
    known = [a for a in answers if a.answer != UNKNOWN]
    options = defaultdict(list)
    for a in known:
        match = OPTION_ID.match(a.id)
        if match:
            options[match.group(1)].append(a)
    implied = {a.id for group in options.values() if len(group) > 1
               for a in group if a.answer != "yes"}

    statements = []  # [quote, answers], in order of first appearance
    for a in known:
        if a.id in implied:
            continue
        for statement in statements:
            if evidence_in_text(a.evidence, statement[0]) or \
                    evidence_in_text(statement[0], a.evidence):
                statement[0] = max(statement[0], a.evidence, key=len)
                statement[1].append(a)
                break
        else:
            statements.append([a.evidence, [a]])
    return "\n".join(
        f'- "{quote}": ' + "; ".join(f"{a.question} {a.answer}" for a in group)
        for quote, group in statements)


def _words(text: str) -> str:
    text = re.sub(r"['‘’]", "", text.lower())
    return " ".join(re.findall(r"[^\W_]+", text))


def evidence_in_text(evidence: str, text: str) -> bool:
    """Whether a quote appears word for word in the text, ignoring case and punctuation."""
    quote = _words(evidence)
    return bool(quote) and f" {quote} " in f" {_words(text)} "


class QuestionAnswerer:
    """Answer questions from a dialogue with one structured LLM call."""

    def __init__(self, llm_client: LLMClient, prompt_config: Mapping[str, Any]):
        self.llm_client = llm_client
        self.config = prompt_config

    def answer(self, dialogue: str,
               questions: Sequence[Question]) -> List[QuestionAnswer]:
        """Answer each question, or return [] if the LLM call fails."""
        if not questions:
            return []

        user_prompt = self.config["user_prompt"].format(
            dialogue=dialogue.strip(),
            questions=format_questions(questions),
        )

        try:
            response = self.llm_client.call_with_schema(
                system_prompt=self.config.get("system_prompt", ""),
                user_prompt=user_prompt,
                schema=build_answer_schema(questions),
                schema_name="questionnaire_answers",
                temperature=0,
            )
        except Exception:
            logger.exception("Questionnaire answering call failed")
            return []

        if response.get("api_failed"):
            logger.warning("Questionnaire answering call failed after retries")
            return []

        return [self._parse(q, response.get(f"q{i}"), dialogue)
                for i, q in enumerate(questions, 1)]

    @staticmethod
    def _parse(question: Question, item: Any, dialogue: str) -> QuestionAnswer:
        item = item if isinstance(item, dict) else {}
        evidence = str(item.get("evidence") or "").strip()
        given = str(item.get("answer") or "").strip().strip(".").lower()
        answer = {a.lower(): a for a in question.allowed_answers}.get(given, UNKNOWN)
        if answer != UNKNOWN and not evidence_in_text(evidence, dialogue):
            logger.warning("Unverified answer %r to %s downgraded to unknown; "
                           "evidence not found: %r", answer, question.id, evidence)
            answer = UNKNOWN
        return QuestionAnswer(id=question.id, question=question.text,
                              answer=answer,
                              evidence=evidence if answer != UNKNOWN else "")
