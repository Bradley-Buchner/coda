"""Build a question bank TSV file used for questionnaire retrieval. The default bank
holds the symptom, background, and sex VA interview questions linked to WHO VA causes in
ProbBase, along with ProbBase's skip rules.
"""

import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from coda.resources import get_resource_path

UNKNOWN = "unknown"
DEFAULT_RESPONSES = ("yes", "no")
DEFAULT_BANK = get_resource_path("questionnaires/who_va_probbase.tsv")
COLUMNS = ["id", "question", "responses", "category", "ask_if", "skip_if"]
# Options of one multi-option item share an id stem: who.va.q:W610148-a, -b, -c.
OPTION_ID = re.compile(r"^(.+)-[a-z]$")

PROBBASE_CATEGORIES = {"M": "symptom", "B": "background", "S": "demographic"}
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
# A newborn's day of death (questions in the bank), each with its lower bound in days
DAY_OF_DEATH = [
    ("who.va.q:W610022-k", 7),   # after the first week, within the first month
    ("who.va.q:W610022-j", 2),   # after 48 hours, within the first week
    ("who.va.q:W610022-i", 1),   # between 24 and 48 hours
    ("who.va.q:W610022-h", 0),   # within 24 hours
]
# Sex and age-at-death questions, whose skip rules say who a question applies to
DEMOGRAPHIC = ("who.va.q:W610019-", "who.va.q:W610022-")
# Options that ProbBase marks as exclusive but that can both be true
OVERLAPPING_OPTIONS = {
    ("who.va.q:W610235-a", "who.va.q:W610235-d"),  # rash on the face / everywhere
    ("who.va.q:W610235-b", "who.va.q:W610235-d"),  # rash on the trunk / everywhere
    ("who.va.q:W610235-c", "who.va.q:W610235-d"),  # rash on the limbs / everywhere
    ("who.va.q:W610260-e", "who.va.q:W610260-f"),  # one leg / one arm paralysed
    ("who.va.q:W610260-f", "who.va.q:W610260-e"),
    ("who.va.q:W610337-b", "who.va.q:W610337-c"),  # gave birth at home / elsewhere (twins)
    ("who.va.q:W610337-c", "who.va.q:W610337-b"),
}
# Follow-ups that don't presuppose their parent (parent: follow-ups): they can be
# true when the parent is unknown, so they aren't held back until it is yes
INDEPENDENT_FOLLOW_UPS = {
    "W610077-o": (  # injury or accident that led to death
        "W610079-o",  # road traffic accident
        "W610082-o",  # non-road transport accident
        "W610083-o",  # fall
        "W610084-o",  # poisoning
        "W610086-o",  # bite or sting of a venomous animal
        "W610087-o",  # non-venomous animal or insect
        "W610089-o",  # burns or fire
        "W610090-o",  # violence (suicide, homicide, abuse)
        "W610095-o",  # force of nature
        "W610096-o",  # electrocution
    ),
    "W610090-o": (  # violence (suicide, homicide, abuse)
        "W610091-o",  # firearm
        "W610092-o",  # stabbed, cut or pierced
        "W610093-o",  # strangled
        "W610094-o",  # blunt force
        "W610099-o",  # self-inflicted injury
    ),
    "W610110-o": ("W610112-o",),  # baby ever breathed -> breathing problem
    "W610114-o": ("W610116-o",),  # baby born dead -> body macerated (soft, skin peeling)
    "W610147-o": ("W610152-o",),  # fever -> night sweats
    "W610159-o": (  # difficulty breathing
        "W610166-o",  # fast breathing
        "W610168-o",  # breathlessness
        "W610172-o",  # chest wall pulled in when breathing
        "W610173-a",  # wheezing or grunting
    ),
    "W610174-o": ("W610176-a",),  # chest pain -> chest pain at least 3 days before death
    "W610181-o": (  # diarrhoea
        "W610183-a",  # at least 4 stools on the worst day
        "W610186-o",  # blood in the stools at any time
        "W610187-o",  # blood in the stools until death
    ),
    "W610188-o": ("W610189-o",),  # vomiting -> vomiting in the week before death
    "W610193-o": ("W610200-o",),  # belly problem -> protruding belly
    "W610215-o": (  # unconscious for at least 24 hours before death
        "W610216-a",  # unconscious for at least 6 hours before death
        "W610217-o",  # unconsciousness started suddenly (within a day)
        "W610218-o",  # unconsciousness continued until death
    ),
    "W610233-o": (  # skin rash
        "W610236-o",  # measles rash
        "W610237-o",  # ever had shingles
        "W610238-o",  # skin flaking off in patches
        "W610239-o",  # skin turning black
        "W610240-o",  # red, swollen skin
    ),
    "W610243-o": ("W610244-o",),  # weight loss -> severely thin or wasted
    "W610253-o": ("W610254-o",),  # any lumps -> lumps or lesions in the mouth
    "W610271-o": ("W610272-o",),  # baby suckled in the first 24 hours -> ever suckled normally
    "W610272-o": ("W610273-o",),  # baby ever suckled normally -> stopped suckling
    "W610281-o": (  # baby became unresponsive or unconscious
        "W610282-o",  # within 24 hours of birth
        "W610283-o",  # more than 24 hours after birth
    ),
    "W610284-o": ("W610285-a",),  # baby became cold to touch -> after 3 days old
    "W610296-o": ("W610301-o",),  # ever menstruated -> heavy vaginal bleeding in the last week
    "W610297-o": ("W610304-o",),  # bleeding between periods -> sharp belly pain before death
    "W610305-o": ("W610309-o",),  # pregnant at death -> pregnant less than 6 months
    "W610306-o": (  # died within 6 weeks of delivery, abortion or miscarriage
        "W610313-o",  # died after delivering a baby
        "W610347-o",  # baby born more than a month early
    ),
    "W610313-o": (  # died after delivering a baby
        "W610337-a",  # gave birth at a health facility
        "W610338-o",  # professional help with the delivery
        "W610342-o",  # normal vaginal delivery
    ),
    "W610316-o": ("W610318-o",),  # gave birth to a live baby -> breastfeeding before death
    "W610325-o": (  # heavy bleeding in pregnancy or soon after delivery
        "W610326-o",  # vaginal bleeding in the first 6 months of pregnancy
        "W610327-o",  # vaginal bleeding in the last 3 months, before labour
        "W610329-o",  # heavy bleeding after delivery or abortion
    ),
    "W610335-o": ("W610336-o",),  # died during an abortion -> within 6 weeks of an abortion
    "W610337-a": (  # gave birth at a health facility
        "W610337-b",  # gave birth at home
        "W610337-c",  # gave birth elsewhere
    ),
    "W610342-o": (  # normal vaginal delivery
        "W610343-o",  # vaginal delivery with forceps or vacuum
        "W610344-o",  # Caesarean section
    ),
    "W610370-o": (  # baby physically abnormal at delivery
        "W610371-o",  # swelling or defect on the back
        "W610372-o",  # very large head
        "W610373-o",  # very small head
    ),
    "W610376-o": ("W610377-o",),  # baby moving before birth -> stopped moving before labour
    "W610418-o": (  # any treatment for the final illness
        "W610419-o",  # oral rehydration salts
        "W610420-o",  # intravenous fluids
        "W610421-o",  # blood transfusion
        "W610422-o",  # tube feeding through the nose
        "W610423-o",  # injectable antibiotics
        "W610424-o",  # antiretroviral therapy
        "W610425-o",  # operation
        "W610427-o",  # discharged from hospital very ill
    ),
    "W610450-o": (  # travelled to a health facility in the final days
        "W610451-o",  # by motorised transport
        "W610452-o",  # problems with admission
        "W610453-o",  # problems with the treatment received
        "W610454-o",  # problems getting medicines or tests
    ),
}
_INDEPENDENT = {("who.va.q:" + child, "who.va.q:" + parent)
                for parent, children in INDEPENDENT_FOLLOW_UPS.items()
                for child in children}

# Some questions need re-wording for clarity (e.g., follow-up questions that lose
# context on their own, unclear sentence subjects, or typos in the source)
PROBBASE_REWORDED = {
    "who.va.q:W610019-a": "Was the deceased male?",
    "who.va.q:W610019-b": "Was the deceased female?",
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


def _age_facts(age_days: Optional[float]) -> dict:
    """Answers the age at death gives: its age band and, for a newborn, the day."""
    if age_days is None:
        return {}
    facts = {qid: "no" for qid, _ in AGE_BANDS + DAY_OF_DEATH}
    facts[next(qid for qid, lower in AGE_BANDS if age_days >= lower)] = "yes"
    if age_days < 28:
        facts[next(qid for qid, lower in DAY_OF_DEATH if age_days >= lower)] = "yes"
    return facts


def _fits(question: Question, facts: dict) -> bool:
    """Whether the deceased's age and sex allow the question."""
    if any(qid.startswith(DEMOGRAPHIC) and facts.get(qid) == answer
           for qid, answer in question.skip_if):
        return False
    return all(facts.get(qid, answer) == answer for qid, answer in question.ask_if
               if qid.startswith(DEMOGRAPHIC))


def gating_parents(question: Question) -> List[Tuple[str, str]]:
    """The parents a follow-up presupposes, with the answer each must have."""
    return [(qid, answer) for qid, answer in question.ask_if
            if not qid.startswith(DEMOGRAPHIC) and (question.id, qid) not in _INDEPENDENT]


def applicable(question: Question, age_days: Optional[float] = None,
               known: Optional[dict] = None) -> bool:
    """Whether a question applies to the deceased under its skip rules. Facts come
    from the age at death and from known yes/no answers. Questions the age itself
    answers are not asked.
    """
    age = _age_facts(age_days)
    facts = {qid: a for qid, a in (known or {}).items() if a in ("yes", "no")}
    facts.update(age)
    if question.id in age or not _fits(question, facts):
        return False
    return all(facts.get(qid, answer) == answer for qid, answer in question.ask_if)


def apply_rules(questions: List[Question], answers: dict,
                age_days: Optional[float] = None) -> dict:
    """Apply the skip rules to yes/no answers (question id -> answer)."""
    by_id = {q.id: q for q in questions}
    age = _age_facts(age_days)
    given = {qid: a for qid, a in answers.items() if a in ("yes", "no") and qid in by_id}
    result = {qid: a for qid, a in given.items() if qid not in age
              and _fits(by_id[qid], {**given, **age})}
    result.update({qid: "yes" for qid, a in age.items() if a == "yes" and qid in by_id})

    implies = defaultdict(list)  # parent -> follow-ups that presuppose it
    for q in questions:
        for parent, answer in gating_parents(q):
            implies[parent].append((q.id, answer))
    changed = True
    while changed:
        changed = False
        facts = {**result, **age}
        for q in questions:
            if q.id in result or not _fits(q, facts):
                continue
            implied = next((answer for child, answer in implies[q.id]
                            if result.get(child) == "yes"), None)
            if implied is None and any(
                    facts.get(qid) == "yes" for qid, a in q.skip_if
                    if a == "yes" and q.group and qid.startswith(q.group + "-")
                    and not qid.startswith(DEMOGRAPHIC)
                    and (q.id, qid) not in OVERLAPPING_OPTIONS):
                implied = "no"
            if implied:
                result[q.id] = implied
                changed = True

    facts = {**result, **age}
    kept = {}
    for qid, answer in result.items():
        q = by_id[qid]
        parents = [(p, a) for p, a in q.ask_if if not p.startswith(DEMOGRAPHIC)]
        if any(p in facts and facts[p] != a for p, a in parents):
            if answer == "yes" and given.get(qid) == "yes":
                kept[qid] = answer
        elif not any(p not in facts for p, _ in gating_parents(q)):
            kept[qid] = answer
    return kept


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
