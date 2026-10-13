"""Build a question bank TSV file used for questionnaire retrieval. The default bank
holds the symptom and background VA interview questions linked to WHO VA causes in
ProbBase, along with ProbBase's skip rules.
"""

import csv
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from coda.resources import get_resource_path

UNKNOWN = "unknown"
DEFAULT_RESPONSES = ("yes", "no")
DEFAULT_BANK = get_resource_path("questionnaires/who_va_probbase.tsv")
COLUMNS = ["id", "question", "responses", "category", "ask_if", "skip_if"]
# Options of one multi-option item share an id stem: who.va.q:W610148-a, -b, -c.
OPTION_ID = re.compile(r"^(.+)-[a-z]$")

PROBBASE_CATEGORIES = {"M": "symptom", "B": "background"}
PROBBASE_RULES = {"ask_if": "ask_if", "dont_ask_if": "skip_if"}
PROBBASE_ANSWERS = {"Y": "yes", "N": "no"}
# ProbBase age bands, oldest first, with each band's lower bound in days
AGE_BANDS = [
    ("who.va.q:W610022-a", 65 * 365.25),  # 65 years or more
    ("who.va.q:W610022-b", 50 * 365.25),
    ("who.va.q:W610022-c", 15 * 365.25),
    ("who.va.q:W610022-d", 5 * 365.25),
    ("who.va.q:W610022-e", 365.25),
    ("who.va.q:W610022-f", 28),
    ("who.va.q:W610022-g", 0),            # neonate
]

# Some questions need re-wording for clarity (e.g., follow-up questions that lose
# context on their own, unclear sentence subjects, or typos in the source)
PROBBASE_REWORDED = {
    "who.va.q:W610216-a": "Was he or she unconscious for at least 6 hours before death?",
    "who.va.q:W610250-a":
        "Did the swelling of the legs or feet last for at least 3 days before death?",
    "who.va.q:W610298-o": "Was her menstrual bleeding excessive?",
    "who.va.q:W610337-a": "Did she give birth at a health facility or clinic?",
    "who.va.q:W610337-b": "Did she give birth at home?",
    "who.va.q:W610337-c":
        "Did she give birth elsewhere (not at a health facility nor at home)?",
    "who.va.q:W610342-o":
        "Did she give birth by normal vaginal delivery, without forceps or vacuum?",
    "who.va.q:W610343-o": "Did she give birth vaginally, with forceps or vacuum?",
    "who.va.q:W610344-o": "Did she give birth by Caesarean section?",
    "who.va.q:W610387-o":
        "Was the baby born by normal vaginal delivery, without forceps or vacuum?",
    "who.va.q:W610388-o": "Was the baby born vaginally, with forceps or vacuum?",
    "who.va.q:W610389-o": "Was the baby born by Caesarean section?",
}

# Change "(s)he" to "he or she" to prevent accidental gendering of questions by
# the sentence embedder.
GENDER_NEUTRAL = [
    (re.compile(r"\b(his/her|her/his)\b", re.IGNORECASE), "his or her"),
    (re.compile(r"\(s\)he\b|\bs\(he\)|\bs/he\b", re.IGNORECASE), "he or she"),
]


def _gender_neutral(text: str) -> str:
    for pattern, replacement in GENDER_NEUTRAL:
        text = pattern.sub(replacement, text)
    return text


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    responses: Tuple[str, ...] = DEFAULT_RESPONSES
    category: str = ""
    ask_if: Tuple[Tuple[str, str], ...] = ()   # (question id, answer) pairs
    skip_if: Tuple[Tuple[str, str], ...] = ()

    @property
    def allowed_answers(self) -> Tuple[str, ...]:
        if UNKNOWN in self.responses:
            return self.responses
        return self.responses + (UNKNOWN,)

    @property
    def group(self) -> Optional[str]:
        """Shared id stem of a multi-option item's options (W610148-a/b/c -> W610148)."""
        match = OPTION_ID.match(self.id)
        return match.group(1) if match else None


def _conditions(text: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    """Parse "id=answer;id=answer" into (id, answer) pairs."""
    return tuple(tuple(c.strip().rsplit("=", 1)) for c in (text or "").split(";")
                 if c.strip())


def applicable(question: Question, age_days: Optional[float] = None,
               known: Optional[dict] = None) -> bool:
    """Whether a question applies to the deceased under its skip rules. Facts come
    from the age at death and from known yes/no answers.
    """
    facts = {}
    if age_days is not None:
        band = next(qid for qid, lower in AGE_BANDS if age_days >= lower)
        facts = {qid: "yes" if qid == band else "no" for qid, _ in AGE_BANDS}
    facts.update({qid: a for qid, a in (known or {}).items() if a in ("yes", "no")})
    if any(facts.get(qid) == answer for qid, answer in question.skip_if):
        return False
    return all(facts.get(qid, answer) == answer for qid, answer in question.ask_if)


def load_question_bank(path: Optional[str] = None) -> List[Question]:
    """Load a question bank TSV. Defaults to the packaged WHO VA bank."""
    path = path or DEFAULT_BANK
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = {"id", "question"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        questions = []
        seen = set()
        for row in reader:
            text = (row["question"] or "").strip()
            if not text:
                continue
            qid = (row["id"] or "").strip()
            if not qid or qid in seen:
                raise ValueError(f"{path}: missing or duplicate id {qid!r}")
            seen.add(qid)
            responses = tuple(r.strip() for r in (row.get("responses") or "").split("|")
                              if r.strip())
            questions.append(Question(
                id=qid,
                text=text,
                responses=responses or DEFAULT_RESPONSES,
                category=(row.get("category") or "").strip(),
                ask_if=_conditions(row.get("ask_if")),
                skip_if=_conditions(row.get("skip_if")),
            ))
    return questions


def question_bank_from_probbase(nodes_file: str, out_file: str,
                                edges_file: Optional[str] = None) -> None:
    """Write the default question bank from the KG's probbase nodes file, with the
    skip rules from its edges file."""
    import pandas as pd

    rules = {column: {} for column in PROBBASE_RULES.values()}
    if edges_file:
        edges = pd.read_csv(edges_file, sep="\t", dtype=str)
        edges = edges[edges[":TYPE"].isin(PROBBASE_RULES)].sort_values([":END_ID", "value"])
        for e in edges.itertuples(index=False):
            start, end, answer, edge_type = e
            column = rules[PROBBASE_RULES[edge_type]]
            column.setdefault(start, []).append(f"{end}={PROBBASE_ANSWERS[answer]}")

    nodes = pd.read_csv(nodes_file, sep="\t", dtype=str)
    nodes = nodes[nodes["samb"].isin(PROBBASE_CATEGORIES)]
    bank = pd.DataFrame({
        "id": nodes["id:ID"],
        "question": nodes["id:ID"].map(PROBBASE_REWORDED).fillna(nodes["name"])
                    .map(_gender_neutral),
        "responses": "",
        "category": nodes["samb"].map(PROBBASE_CATEGORIES),
        **{column: nodes["id:ID"].map(lambda qid, c=conditions: ";".join(c.get(qid, [])))
           for column, conditions in rules.items()},
    })
    bank.sort_values("id").to_csv(out_file, sep="\t", index=False,
                                  columns=COLUMNS, lineterminator="\n")


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: python -m coda.questionnaire.bank "
                 "<probbase_nodes.tsv.gz> <output.tsv> [probbase_edges.tsv.gz]")
    edges_file = sys.argv[3] if len(sys.argv) == 4 else \
        sys.argv[1].replace("_nodes", "_edges")
    question_bank_from_probbase(sys.argv[1], sys.argv[2], edges_file)


if __name__ == "__main__":
    main()
