"""Answer the WHO-form bank's questions from the dialogue with an LLM.

Questions are asked in rounds, as allowed by the form's skip logic (see who_form.Interview).

A question can be answered by more than yes/no; it can take one or more options from a list,
a number (exact, or a range marked stated or implied) or a short text. An answer whose
evidence quote isn't in the dialogue is downgraded to "unknown".
"""
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from coda.llm_api.client import LLMClient
from coda.questionnaire.answerer import QuestionAnswer, evidence_in_text
from coda.questionnaire.bank import Question, _age_facts
from coda.questionnaire.who_form import (PREFIX, UNKNOWN, Interview, WhoForm, WhoQuestion,
                                         items_from)

logger = logging.getLogger(__name__)

MAX_ENUM_VALUES = 1000
NONE_OF_THESE = "None of these"
STILLBIRTH = {"Id10104": "no", "Id10109": "no", "Id10110": "no", "Id10114": "yes"}  # auto-answers when stillbirth
SEX_ITEMS = {"male": {"W610019-a": "yes", "W610019-b": "no"},
             "female": {"W610019-a": "no", "W610019-b": "yes"},
             "undetermined": {"W610019-a": "no", "W610019-b": "no"}}  # auto-answers when sex is known


@dataclass(frozen=True)
class WhoAnswer:
    id: str
    question: str
    answer: Any
    evidence: str
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _labels(q: WhoQuestion) -> List[str]:
    return [NONE_OF_THESE if q.kind == "multi" and name.lower() == "no" else label
            for name, label in q.choices]


def _wording(q: WhoQuestion, newborn: bool = False) -> Tuple[str, str]:
    """The question's text and hint note when there is one."""
    if newborn and q.text_newborn:
        return q.text_newborn, q.note_newborn
    return q.text, q.note


def _answer_hint(q: WhoQuestion) -> str:
    """The kind of answer a question takes, shown after it."""
    if q.kind == "yesno":
        return "yes | no | unknown"
    if q.kind == "choice":
        return "one of: " + " | ".join(_labels(q)) + " | unknown"
    if q.kind == "multi":
        return "any of: " + " | ".join(_labels(q)) + " | unknown"
    if q.kind == "number":
        return f"a number of {' or '.join(q.units)}, or a range | unknown"
    return "a short text | unknown"


def format_who_questions(questions: Sequence[WhoQuestion], newborn: bool = False) -> str:
    """Formats the questions as numbered prompt lines with the kind of answer each takes,
    followed by its WHO hint note when it has one."""
    lines = []
    for i, q in enumerate(questions, 1):
        text, note = _wording(q, newborn)
        lines.append(f"q{i}: {text} [{_answer_hint(q)}]")
        if note:
            lines.append(f"    WHO hint note: {note}")
    return "\n".join(lines)


def shown(q: WhoQuestion, answer: Any) -> str:
    """An answer as the prompt shows it, e.g. "Face, Everywhere" or "2 to 4 days"."""
    if answer == UNKNOWN or q.kind in ("yesno", "text"):
        return str(answer)
    labels = dict(zip((name for name, _ in q.choices), _labels(q)))
    if q.kind == "choice":
        return labels.get(answer, answer)
    if q.kind == "multi":
        return ", ".join(labels.get(a, a) for a in answer)
    low, high, unit = answer["low"], answer["high"], answer["unit"]
    if low == high:
        return f"{low:g} {unit}"
    if low is None:
        return f"at most {high:g} {unit}"
    if high is None:
        return f"at least {low:g} {unit}"
    return f"{low:g} to {high:g} {unit}"


def recheck_note(check: Mapping[str, Any], questions: Sequence[WhoQuestion],
                 answers: Mapping[str, Any]) -> str:
    """The note that follows questions re-asked by a go-back check."""
    earlier = "; ".join(f"q{i}: {shown(q, answers[q.id])}"
                        for i, q in enumerate(questions, 1))
    return ("Check: the earlier answers to these questions can't all be true. "
            f"The WHO form's check says: \"{check['message']}\" "
            f"Earlier answers: {earlier}. Answer these questions again.")


def _answer_schema(q: WhoQuestion) -> Dict[str, Any]:
    if q.kind == "yesno":
        return {"type": "string", "enum": ["yes", "no", UNKNOWN]}
    if q.kind == "choice":
        return {"type": "string", "enum": _labels(q) + [UNKNOWN]}
    if q.kind == "multi":
        return {"type": "array", "items": {"type": "string", "enum": _labels(q) + [UNKNOWN]}}
    if q.kind == "number":
        return {"type": "object",
                "properties": {"low": {"type": ["number", "null"]},
                               "high": {"type": ["number", "null"]},
                               "unit": {"type": "string", "enum": list(q.units)},
                               "basis": {"type": "string",
                                         "enum": ["stated", "implied", UNKNOWN]}},
                "required": ["low", "high", "unit", "basis"],
                "additionalProperties": False}
    return {"type": "string"}


def build_who_schema(questions: Sequence[WhoQuestion]) -> Dict[str, Any]:
    """Build the JSON schema of an {evidence, answer} entry per question."""
    properties = {
        f"q{i}": {
            "type": "object",
            "properties": {"evidence": {"type": "string"}, "answer": _answer_schema(q)},
            "required": ["evidence", "answer"],
            "additionalProperties": False,
        }
        for i, q in enumerate(questions, 1)
    }
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _enum_size(q: WhoQuestion) -> int:
    return {"yesno": 3, "choice": len(q.choices) + 1, "multi": len(q.choices) + 1,
            "number": len(q.units) + 3}.get(q.kind, 0)


def _bound(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _parse_number(q: WhoQuestion, given: Any) -> Any:
    if not isinstance(given, dict) or given.get("basis") not in ("stated", "implied"):
        return UNKNOWN
    low, high = _bound(given.get("low")), _bound(given.get("high"))
    unit = given.get("unit")
    if unit not in q.units:
        unit = q.units[0] if len(q.units) == 1 else None
    if unit is None or (low is None and high is None):
        return UNKNOWN
    if low is not None and high is not None and low > high:
        low, high = high, low
    return {"low": low, "high": high, "unit": unit, "basis": given["basis"]}


def parse_answer(q: WhoQuestion, item: Any, dialogue: str, text: str) -> WhoAnswer:
    """Read one question's entry in the LLM's response. An answer whose evidence isn't
    in the dialogue is downgraded to unknown."""
    item = item if isinstance(item, dict) else {}
    evidence = str(item.get("evidence") or "").strip()
    given = item.get("answer")
    names = {label.lower(): name for (name, _), label in zip(q.choices, _labels(q))}
    if q.kind == "yesno":
        given = str(given or "").strip().strip(".").lower()
        answer = given if given in ("yes", "no") else UNKNOWN
    elif q.kind == "choice":
        answer = names.get(str(given or "").strip().lower(), UNKNOWN)
    elif q.kind == "multi":
        chosen = [str(g).strip().lower() for g in given] if isinstance(given, list) else []
        picked = [names[c] for c in chosen if c in names]
        answer = picked if picked and UNKNOWN not in chosen else UNKNOWN
    elif q.kind == "number":
        answer = _parse_number(q, given)
    else:
        given = str(given or "").strip()
        answer = given if given and given.lower() != UNKNOWN else UNKNOWN
    if answer != UNKNOWN and not evidence_in_text(evidence, dialogue):
        logger.warning("Unverified answer %r to %s downgraded to unknown; "
                       "evidence not found: %r", answer, q.id, evidence)
        answer = UNKNOWN
    return WhoAnswer(id=q.id, question=text, answer=answer,
                     evidence=evidence if answer != UNKNOWN else "")


class WhoAnswerer:
    """For answering the WHO-form bank's questions from a dialogue, in rounds of LLM calls."""

    def __init__(self, llm_client: LLMClient, prompt_config: Mapping[str, Any]):
        self.llm_client = llm_client
        self.config = prompt_config

    def answer(self, dialogue: str, questions: Sequence[WhoQuestion], newborn: bool = False,
               note: str = "") -> Optional[List[WhoAnswer]]:
        """Answers the questions in one call."""
        if not questions:
            return []
        listed = format_who_questions(questions, newborn)
        if note:
            listed += "\n" + note
        user_prompt = self.config["user_prompt"].format(dialogue=dialogue.strip(),
                                                        questions=listed)
        try:
            response = self.llm_client.call_with_schema(
                system_prompt=self.config.get("system_prompt", ""),
                user_prompt=user_prompt,
                schema=build_who_schema(questions),
                schema_name="questionnaire_answers",
                temperature=0,
            )
        except Exception:
            logger.exception("Questionnaire answering call failed")
            return None
        if response.get("api_failed"):
            logger.warning("Questionnaire answering call failed after retries")
            return None
        return [parse_answer(q, response.get(f"q{i}"), dialogue, _wording(q, newborn)[0])
                for i, q in enumerate(questions, 1)]

    def _answer_in_batches(self, dialogue: str, questions: Sequence[WhoQuestion],
                           newborn: bool) -> Optional[List[WhoAnswer]]:
        """Answer the questions in as few calls as the enum limit allows."""
        batches, size = [[]], 0
        for q in questions:
            if batches[-1] and size + _enum_size(q) > MAX_ENUM_VALUES:
                batches.append([])
                size = 0
            batches[-1].append(q)
            size += _enum_size(q)
        answers = []
        for batch in batches:
            got = self.answer(dialogue, batch, newborn)
            if got is None:
                return None
            answers += got
        return answers

    def answer_form(self, dialogue: str, form: WhoForm, age_days: Optional[float] = None,
                    sex: Optional[str] = None, stillbirth: bool = False
                    ) -> Tuple[List[WhoAnswer], Interview, bool]:
        """Ask the form's questions in rounds until none are left."""
        prefill = STILLBIRTH if stillbirth else {}
        if stillbirth and age_days is None:
            age_days = 0.0
        interview = Interview(form, age_days, sex.strip().lower() if sex else None, prefill)
        answers = {qid: WhoAnswer(qid, form.by_id[qid].text, a, "", source="profile")
                   for qid, a in prefill.items()}

        def finish(complete):
            return [answers[qid] for qid in interview.answers], interview, complete

        checked = set()
        while True:
            questions = interview.next_round()
            if not questions:
                return finish(True)
            got = self._answer_in_batches(dialogue, questions, interview.newborn())
            if got is None:
                return finish(False)
            for a in got:
                interview.record(a.id, a.answer)
                answers[a.id] = a
            for check in form.checks:
                if check["id"] in checked or not interview.fires(check):
                    continue
                checked.add(check["id"])
                if not self._reask(dialogue, interview, check, answers):
                    return finish(False)

    def _reask(self, dialogue: str, interview: Interview, check: Mapping[str, Any],
               answers: Dict[str, WhoAnswer]) -> bool:
        """Re-ask questions if needed."""
        questions = [interview.form.by_id[qid] for qid in check["reask"]
                     if qid in interview.answers and qid not in interview.prefilled]
        if not questions:
            return True
        before = {q.id: interview.answers[q.id] for q in questions}
        got = self.answer(dialogue, questions, interview.newborn(),
                          recheck_note(check, questions, before))
        if got is None:
            return False
        answers.update({a.id: a for a in got})
        interview.revise({a.id: a.answer for a in got})
        interview.rechecked.append({"check": check["id"], "before": before,
                                    "after": {a.id: a.answer for a in got},
                                    "resolved": not interview.fires(check)})
        return True


def _age_items(interview: Interview) -> Dict[str, str]:
    """The age band and newborn day-of-death items that the age at death gives."""
    low, high = interview.age_range()
    if (low is None and high is None) or (low is not None and high is not None and low > high):
        return {}
    oldest = 200 * 365.25
    young = _age_facts(low or 0.0)
    old = _age_facts(oldest if high is None else min(high, oldest))
    return {qid[len(PREFIX):]: young[qid] if young[qid] == old[qid] else UNKNOWN
            for qid in young}


def who_record(answers: Sequence[WhoAnswer], interview: Interview,
               probbase_bank: Sequence[Question]) -> Dict[str, Any]:
    """The WHO answers and the probbase items they convert to."""
    by_id = {a.id: a for a in answers}
    profile = SEX_ITEMS.get(interview.profile_sex, {})
    converted = {}
    for item, (value, qid) in items_from(interview.form, {a.id: a.answer for a in answers},
                                         profile).items():
        a = by_id[qid]
        source = "profile" if a.source == "profile" else f"who_form:{qid}"
        converted[item] = (value, a.evidence if value != UNKNOWN else "", source)
    converted.update({item: (v, "", "age") for item, v in _age_items(interview).items()})
    converted.update({item: (v, "", "profile") for item, v in profile.items()})
    items = []
    for q in probbase_bank:
        item = q.id[len(PREFIX):]
        if item in converted:
            answer, evidence, source = converted[item]
            items.append(QuestionAnswer(q.id, q.text, answer, evidence, source))
    return {"answers": [a.to_dict() for a in answers], "items": [i.to_dict() for i in items],
            "rounds": interview.rounds, "rechecked": interview.rechecked,
            "skipped": sorted(interview.skipped)}
