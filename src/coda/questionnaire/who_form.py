"""Parses and applies the questionnaire question-asking logic for the WHO 2016 VA form,
used by CODA's `who_form` questionnaire strategy.

Each question in the bank (built by who_bank.py) keeps the form's ODK "relevant"
expressions, specifying what other answers trigger it to be asked.

Additionally...
- An unasked question is blank, and "unknown" corresponds to "doesn't know" in the form.
- Some questions were decided as `yes_only` gates (e.g. injury); for these, "unknown" counts as unasked
- Age and sex are determined from the case profile, or otherwise asked as part of the questionnaire; this can result in
 their answers being left as "unknown".
"""
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from coda.questionnaire.bank import UNKNOWN
from coda.resources import get_resource_path

DEFAULT_WHO_BANK = get_resource_path("questionnaires/who_va_2016.json")
PREFIX = "who.va.q:"
KINDS = ("yesno", "choice", "multi", "number", "text")
SEX_QUESTION = "Id10019"
AGE_QUESTIONS = ("age_group", "age")
AGE_FIELDS = {"isNeonatal", "isChild", "isAdult", "ageInMonthsByYear"}
DAYS_PER = {"minutes": 1 / 1440, "hours": 1 / 24, "days": 1.0, "weeks": 7.0,
            "months": 30.0, "years": 365.0}
AGE_GROUPS = {"neonate": (0.0, 27.999, (5.0,)),
              "child": (28.0, 12 * 365.25 - 0.001, (180.0, 5 * 365.25)),
              "adult": (12 * 365.25, math.inf, (30 * 365.25,))} # Age group -> (lower bound, upper bound, stand-in when exact age is unknown)

# ODK expressions: and, or, comparisons, selected(), not() and string-length()
TOKEN = re.compile(r"\s*(\$\{[^}]+\}|'[^']*'|\"[^\"]*\"|\d+(?:\.\d+)?|!=|<=|>=|=|<|>|"
                   r"\(|\)|,|[A-Za-z_][\w\-]*)")
FUNCTIONS = {"selected": 2, "not": 1, "string-length": 1}
COMPARISONS = ("=", "!=", "<", ">", "<=", ">=")


def parse(expr: str):
    """Parse an ODK expression into a tree of tuples."""
    tokens, pos = [], 0
    while expr[pos:].strip():
        match = TOKEN.match(expr, pos)
        if not match:
            raise ValueError(f"cannot read {expr[pos:pos + 20]!r} in {expr!r}")
        tokens.append(match.group(1))
        pos = match.end()
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else None

    def take(expected=None):
        nonlocal i
        token = peek()
        if token is None or expected not in (None, token):
            raise ValueError(f"expected {expected or 'more'}, got {token} in {expr!r}")
        i += 1
        return token

    def or_expr():
        tree = and_expr()
        while peek() == "or":
            take()
            tree = ("or", tree, and_expr())
        return tree

    def and_expr():
        tree = comparison()
        while peek() == "and":
            take()
            tree = ("and", tree, comparison())
        return tree

    def comparison():
        left = atom()
        if peek() in COMPARISONS:
            return ("cmp", take(), left, atom())
        return left

    def atom():
        token = take()
        if token == "(":
            tree = or_expr()
            take(")")
            return tree
        if token.startswith("${"):
            return ("field", token[2:-1])
        if token[0] in "'\"":
            return ("str", token[1:-1])
        if token[0].isdigit():
            return ("num", float(token))
        if token in FUNCTIONS:
            take("(")
            args = [or_expr()]
            while peek() == ",":
                take()
                args.append(or_expr())
            take(")")
            if len(args) != FUNCTIONS[token]:
                raise ValueError(f"{token}() takes {FUNCTIONS[token]} argument(s) in {expr!r}")
            return ("call", token, tuple(args))
        raise ValueError(f"unexpected {token!r} in {expr!r}")

    tree = or_expr()
    if i != len(tokens):
        raise ValueError(f"trailing tokens in {expr!r}")
    return tree


def refs(tree) -> Set[str]:
    """The fields an expression refers to."""
    if tree[0] == "field":
        return {tree[1]}
    found = set()
    for part in tree[2] if tree[0] == "call" else tree[1:]:
        if isinstance(part, tuple):
            found |= refs(part)
    return found


def _compare(op: str, x, y) -> bool:
    return {"=": x == y, "!=": x != y, "<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y}[op]


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _true(value) -> bool:
    if isinstance(value, float):
        return not math.isnan(value) and value != 0
    return bool(value)


def evaluate(tree, value) -> Any:
    """Evaluate a tree. value(field) gives a string ('' if blank) or a number."""
    kind = tree[0]
    if kind == "field":
        return value(tree[1])
    if kind in ("str", "num"):
        return tree[1]
    if kind == "or":
        return _true(evaluate(tree[1], value)) or _true(evaluate(tree[2], value))
    if kind == "and":
        return _true(evaluate(tree[1], value)) and _true(evaluate(tree[2], value))
    if kind == "cmp":
        op, a, b = tree[1], evaluate(tree[2], value), evaluate(tree[3], value)
        if isinstance(a, float) or isinstance(b, float):
            x, y = _float(a), _float(b)
            if math.isnan(x) or math.isnan(y):
                return op == "!="
            return _compare(op, x, y)
        return op in ("=", "!=") and _compare(op, str(a), str(b))
    name, args = tree[1], [evaluate(a, value) for a in tree[2]]
    if name == "selected":
        return str(args[1]) in str(args[0]).split()
    if name == "not":
        return not _true(args[0])
    return float(len(str(args[0])))


@dataclass(frozen=True)
class WhoQuestion:
    id: str
    kind: str
    text: str
    relevant: Tuple[str, ...] = ()
    text_newborn: str = ""
    note: str = ""
    note_newborn: str = ""
    choices: Tuple[Tuple[str, str], ...] = ()
    units: Tuple[str, ...] = ()
    base_unit: str = ""
    converts: Tuple[Mapping[str, Any], ...] = ()


class WhoForm:
    """The bank's questions in the order they appear in the form, along with its go-back
    checks and yes-only gates."""

    def __init__(self, questions: Sequence[WhoQuestion],
                 checks: Sequence[Mapping[str, Any]] = (), yes_only: Sequence[str] = ()):
        self.questions = list(questions)
        self.checks = list(checks)
        self.yes_only = set(yes_only)
        self.by_id = {q.id: q for q in self.questions}
        self.trees = {}
        self.waits = {}
        for q in self.questions:
            fields = set()
            for expr in q.relevant:
                fields |= refs(self.tree(expr))
            waits = fields - AGE_FIELDS
            if fields & AGE_FIELDS:
                waits |= set(AGE_QUESTIONS)
            bad = (fields - AGE_FIELDS - set(self.by_id)) | (waits - set(self.waits))
            if bad:
                raise ValueError(f"{q.id}: skip logic refers to unknown or later fields "
                                 f"{sorted(bad)}")
            self.waits[q.id] = waits

    def tree(self, expr: str):
        """The parsed expression, parsed once."""
        if expr not in self.trees:
            self.trees[expr] = parse(expr)
        return self.trees[expr]


def load_who_form(path: Optional[str] = None) -> WhoForm:
    """Load a bank built by who_bank.py. Defaults to the packaged bank."""
    with open(path or DEFAULT_WHO_BANK) as fh:
        data = json.load(fh)
    questions = []
    for q in data["questions"]:
        if q["kind"] not in KINDS:
            raise ValueError(f"{q['id']}: unknown kind {q['kind']!r}")
        questions.append(WhoQuestion(
            id=q["id"], kind=q["kind"], text=q["text"],
            relevant=tuple(q.get("relevant", ())),
            text_newborn=q.get("text_newborn", ""), note=q.get("note", ""),
            note_newborn=q.get("note_newborn", ""),
            choices=tuple(tuple(c) for c in q.get("choices", ())),
            units=tuple(q.get("units", ())), base_unit=q.get("base_unit", ""),
            converts=tuple(q.get("converts", ()))))
    return WhoForm(questions, data.get("checks", ()), data.get("yes_only", ()))


def number_in(answer: Mapping[str, Any],
              unit: str) -> Tuple[Optional[float], Optional[float]]:
    """A number answer's (low, high) bounds converted to `unit`"""
    scale = 1.0
    if answer.get("unit") in DAYS_PER and unit in DAYS_PER:
        scale = DAYS_PER[answer["unit"]] / DAYS_PER[unit]
    low, high = answer.get("low"), answer.get("high")
    return (None if low is None else float(low) * scale,
            None if high is None else float(high) * scale)


class Interview:
    """The answers so far, and which questions the form asks next.

    An answer is "yes"/"no", a choice name, a list of choice names, a number
    {"low", "high", "unit", "basis"}, text, or "unknown".
    """

    def __init__(self, form: WhoForm, age_days: Optional[float] = None,
                 sex: Optional[str] = None, prefill: Optional[Mapping[str, Any]] = None):
        self.form = form
        self.profile_age = age_days if age_days is not None and age_days >= 0 else None
        self.profile_sex = sex
        self.answers: Dict[str, Any] = dict(prefill or {})
        self.prefilled = set(prefill or ())
        self.rounds: List[List[str]] = []
        self.rechecked: List[Dict[str, Any]] = []
        self.skipped: Set[str] = set()

    def age_range(self) -> Tuple[Optional[float], Optional[float]]:
        """The age at death in days as (low, high); None where unknown."""
        if self.profile_age is not None:
            return self.profile_age, self.profile_age
        age = self.answers.get("age")
        low, high = number_in(age, "days") if isinstance(age, dict) else (None, None)
        group = AGE_GROUPS.get(self.answers.get("age_group"))
        if group:
            low = group[0] if low is None else max(low, group[0])
            high = group[1] if high is None else min(high, group[1])
        return low, high

    def _ages(self) -> List[float]:
        if self.profile_age is not None:
            return [self.profile_age]
        age = self.answers.get("age")
        low, high = number_in(age, "days") if isinstance(age, dict) else (None, None)
        if low is not None and low == high:
            return [low]
        group = self.answers.get("age_group")
        if group in AGE_GROUPS:
            return list(AGE_GROUPS[group][2])
        low = -math.inf if low is None else low
        high = math.inf if high is None else high
        ages = []
        for youngest, oldest, tries in AGE_GROUPS.values():
            if low <= oldest and high >= youngest:
                ages += tries
        return ages

    def _sexes(self) -> List[str]:
        sex = self.profile_sex or self.answers.get(SEX_QUESTION)
        return [sex] if sex in ("female", "male", "undetermined") else ["female", "male"]

    def newborn(self) -> bool:
        return all(age <= 27 for age in self._ages())

    def _value(self, age: float, sex: str):
        """value(field) for evaluating expressions at one age and sex."""
        months = int(age / 365.25) * 12 + int(math.fmod(age, 365.25) / 30.4375)
        facts = {"isNeonatal": "1" if age <= 27 else "0",
                 "isChild": "1" if 27 < age < 12 * 365.25 else "0",
                 "isAdult": "1" if age >= 12 * 365.25 else "0",
                 "ageInMonthsByYear": float(months), SEX_QUESTION: sex}

        def value(field):
            if field in facts:
                return facts[field]
            answer = self.answers.get(field)
            if answer is None:
                return ""
            q = self.form.by_id[field]
            if answer == UNKNOWN:
                if field in self.form.yes_only:
                    return ""
                return 99.0 if q.kind == "number" else "dk"
            if q.kind == "multi":
                return " ".join(answer)
            if q.kind == "number":
                low, high = number_in(answer, q.base_unit or answer["unit"])
                return low if low is not None and low == high else 99.0
            return answer
        return value

    def asks(self, q: WhoQuestion) -> bool:
        """Whether the form asks the question."""
        if q.id in AGE_QUESTIONS:
            return self.profile_age is None
        if q.id == SEX_QUESTION and self.profile_sex:
            return False
        if not q.relevant:
            return True
        values = [self._value(age, sex) for age in self._ages() for sex in self._sexes()]
        return any(_true(evaluate(self.form.tree(e), value))
                   for value in values for e in q.relevant)

    def _decide(self) -> List[WhoQuestion]:
        """Decide each question in form order. A question waits while one it refers to is
        still to be answered.
        """
        waiting, to_ask, self.skipped = set(), [], set()
        for q in self.form.questions:
            if q.id in self.prefilled:
                continue
            if self.form.waits[q.id] & waiting:
                waiting.add(q.id)
            elif not self.asks(q):
                self.skipped.add(q.id)
                self.answers.pop(q.id, None)
            elif q.id not in self.answers:
                to_ask.append(q)
                waiting.add(q.id)
        return to_ask

    def next_round(self) -> List[WhoQuestion]:
        questions = self._decide()
        if questions:
            self.rounds.append([q.id for q in questions])
        return questions

    def record(self, qid: str, answer: Any) -> None:
        self.answers[qid] = answer

    def revise(self, answers: Mapping[str, Any]) -> None:
        """Replace answers after a re-ask and decide the skip logic again."""
        self.answers.update(answers)
        self._decide()

    def fires(self, check: Mapping[str, Any]) -> bool:
        """Whether a go-back check's condition holds."""
        value = self._value(self._ages()[0], self._sexes()[0])
        return _true(evaluate(self.form.tree(check["condition"]), value))


def _number_item(answer: Mapping[str, Any], rule: Mapping[str, Any], unit: str) -> str:
    """`yes` if the whole range meets the rule, `no` if none of it does, else unknown."""
    low, high = number_in(answer, unit)
    low = -math.inf if low is None else low
    high = math.inf if high is None else high
    op, v = rule["op"], rule.get("value")
    if op == "range":
        start, end = rule["range"]
        inside = start <= low and high < end
        outside = high < start or low >= end
    elif op == "=":
        inside = low == high == v
        outside = not low <= v <= high
    else:
        inside = _compare(op, low, v) and _compare(op, high, v)
        outside = not (_compare(op, low, v) or _compare(op, high, v) or low < v < high)
    return "yes" if inside else "no" if outside else UNKNOWN


def items_from(form: WhoForm, answers: Mapping[str, Any],
               known: Optional[Mapping[str, str]] = None) -> Dict[str, Tuple[str, str]]:
    """Convert answers to probbase items: item id -> (yes/no/unknown, question id).
    """
    rank = {"yes": 2, "no": 1, UNKNOWN: 0}
    items: Dict[str, Tuple[str, str]] = {}
    for qid, answer in answers.items():
        q = form.by_id.get(qid)
        if q is None:
            continue
        for rule in q.converts:
            if answer == UNKNOWN:
                value = UNKNOWN
            elif q.kind == "yesno":
                value = answer if answer in ("yes", "no") else UNKNOWN
            elif q.kind == "number":
                value = _number_item(answer, rule, q.base_unit)
            elif q.kind == "choice":
                value = "yes" if answer in rule["yes_if"] else "no"
            elif q.kind == "multi":
                chosen, yes = set(answer), set(rule["yes_if"])
                together = set(rule.get("yes_if_all", ()))
                if rule.get("mode") == "only":
                    hit = bool(chosen) and chosen <= yes
                else:
                    hit = bool(chosen & yes)
                value = "yes" if hit or (together and together <= chosen) else "no"
            else:
                continue
            item = rule["item"]
            if item not in items or rank[value] > rank[items[item][0]]:
                items[item] = (value, qid)
    for q in form.questions:
        for rule in q.converts:
            need, item = rule.get("needs"), rule["item"]
            if need and item in items and items[item][0] == "yes":
                other = (known or {}).get(need) or items.get(need, (UNKNOWN,))[0]
                if other != "yes":
                    items[item] = ("no" if other == "no" else UNKNOWN, items[item][1])
    return items
