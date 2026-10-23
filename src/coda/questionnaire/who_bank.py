"""Build the WHO-form question bank used by the `who_form` questionnaire strategy.

The bank asks the WHO 2016 VA form's own questions (ODK XLSForm v1.5.3) with the form's
skip logic, as reviewed for CODA:
- WHO's wording with a few listed changes
- durations, counts, choices and multi-selects are asked as such instead of probbase's yes/no items

Each question also lists the probbase items its answer converts to.
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from typing import List

import pandas as pd

from coda.questionnaire.bank import load_question_bank
from coda.questionnaire.who_form import DEFAULT_WHO_BANK, PREFIX, parse, refs

SKIP_CHOICES = {"dk", "ref", "DK", "Ref"}
UNITS = ("minutes", "hours", "days", "weeks", "months", "years")

# Numbers: stem -> (base unit, {item suffix: (op, value) or ("range", low, high)})
NUMBERS = {
    "W610106": ("minutes", {"a": (">", 5)}),
    "W610108": ("hours", {"a": (">", 24)}),
    "W610120": ("days", {"a": ("<", 21), "b": (">=", 21)}),
    "W610148": ("days", {"a": ("<", 7), "b": ("range", 7, 14), "c": (">=", 14)}),
    "W610154": ("days", {"a": ("<", 21), "b": (">=", 21)}),
    "W610161": ("days", {"a": (">=", 3)}),
    "W610167": ("days", {"a": ("<", 14), "b": (">=", 14)}),
    "W610169": ("days", {"a": ("<", 14), "b": (">=", 14)}),
    "W610176": ("days", {"a": (">=", 3)}),
    "W610178": ("minutes", {"a": (">=", 30)}),
    "W610182": ("days", {"a": ("<", 14), "b": ("range", 14, 28), "c": (">=", 28)}),
    "W610183": ("stools", {"a": (">=", 4)}),
    "W610184": ("days", {"a": (">=", 3)}),
    "W610190": ("days", {"o": (">=", 3)}),
    "W610197": ("days", {"a": ("<", 14), "b": (">=", 14)}),
    "W610201": ("days", {"a": ("<", 14), "b": (">=", 14)}),
    "W610205": ("days", {"a": ("<", 14), "b": (">=", 14)}),
    "W610209": ("days", {"a": ("<", 7), "b": (">=", 7)}),
    "W610211": ("days", {"a": (">=", 7)}),
    "W610213": ("months", {"o": (">=", 3)}),
    "W610216": ("hours", {"a": (">=", 6)}),
    "W610221": ("minutes", {"a": ("<", 10), "b": (">=", 10)}),
    "W610232": ("days", {"a": (">=", 14)}),
    "W610234": ("days", {"a": ("<", 7), "b": (">=", 7)}),
    "W610248": ("days", {"a": (">=", 7)}),
    "W610250": ("days", {"a": (">=", 3)}),
    "W610262": ("days", {"a": (">=", 7)}),
    "W610266": ("days", {"a": (">=", 21)}),
    "W610274": ("days", {"a": (">=", 1)}),
    "W610285": ("days", {"a": (">", 3)}),
    "W610303": ("weeks", {"a": (">=", 4)}),
    "W610309": ("months", {"o": ("<", 6)}),
    "W610319": ("births", {"a": ("=", 0), "b": (">=", 4)}),
    "W610332": ("hours", {"a": (">", 24)}),
    "W610358": ("months", {"a": ("<", 12)}),
    "W610367": ("months", {"a": (">=", 9), "b": ("=", 8), "c": ("<", 8)}),
    "W610382": ("hours", {"a": (">", 24)}),
    "W610394": ("births", {"a": ("=", 0), "b": (">=", 4)}),
    "W610415": ("cigarettes", {"a": (">=", 10)}),
}
# Single choices: stem -> (field, {item suffix: choices that make it yes})
CHOICES = {
    "W610004": ("Id10004", {"a": ["wet"], "b": ["dry"]}),
    "W610019": ("Id10019", {"a": ["male"], "b": ["female"]}),
    "W610059": ("Id10059", {"o": ["married"]}),
    "W610150": ("Id10150", {"a": ["severe"]}),
    "W610151": ("Id10151", {"a": ["continuous"]}),
    "W610165": ("Id10165", {"a": ["continuous"]}),
    "W610199": ("Id10199", {"a": ["upper_abdomen", "upper_lower_abdomen"],
                            "b": ["lower_abdomen", "upper_lower_abdomen"]}),
    "W610203": ("Id10203", {"a": ["rapidly"]}),
    "W610263": ("Id10263", {"a": ["solids", "both"], "b": ["liquids", "both"]}),
    "W610337": ("Id10337", {"a": ["hospital", "other_health_facility"], "b": ["home"],
                            "c": ["on_route_to_hospital_or_facility", "other"]}),
    "W610355": ("Id10355", {"a": ["first"]}),
    "W610357": ("Id10357", {"o": ["during_delivery", "after_delivery"]}),
    "W610360": ("Id10360", {"a": ["hospital", "other_health_facility"], "b": ["home"],
                            "c": ["on_route_to_hospital_or_facility", "other"]}),
    "W610385": ("Id10385", {"a": ["green_or_brown"]}),
    "W610414": ("Id10414", {"a": ["chewing_tobacco"]}),
}
# Multi-selects: stem -> (field, {item suffix: (mode, choices[, choices that together
# count])}). Mode "any": yes if any of the choices is picked; "only": yes if only they are.
MULTI = {
    "W610173": ("Id10173_nc", {"a": ("any", ["wheezing", "grunting"])}),
    "W610235": ("Id10235", {"a": ("any", ["face", "everywhere"]),
                            "b": ("any", ["trunk", "everywhere"]),
                            "c": ("any", ["extremities", "everywhere"]),
                            "d": ("any", ["everywhere"], ["face", "trunk", "extremities"])}),
    "W610260": ("Id10260", {"a": ("only", ["right_side"]),
                            "b": ("only", ["left_side"]),
                            "c": ("only", ["lower_part_of_body"]),
                            "d": ("only", ["upper_part_of_body"]),
                            "e": ("only", ["one_leg_only"]),
                            "f": ("only", ["one_arm_only"]),
                            "g": ("only", ["whole_body"])}),
}
# Items that are only yes when another item is yes too
NEEDS = {"W610197-a": "W610195-o", "W610197-b": "W610195-o", "W610059-o": "W610019-b"}
# WHO asks adults a separate yes/no question about wheezing
ADULT_VERSION = {"W610173": (
    "Id10173_a", "During the illness that led to death did he or she have wheezing?")}

# Wording changes. A yes/no question set to None keeps the probbase bank's wording.
QUESTION_WORDING = {
    "W610165": "Was the difficulty breathing continuous or on and off?",
    "W610161": "For how long did the difficulty breathing last?",
    "W610319": ("How many births, including stillbirths, did she (the mother) have before "
                "this baby?"),
    "W610173": ("During the illness that led to death did his or her breathing sound like "
                "any of the following: stridor, grunting or wheezing?"),
}
YESNO_WORDING = {
    "W610079-o": "Was the injury from a road traffic accident?",
    "W610096-o": "Was the injury from electrocution?",
    "W610086-o": None,
    "W610089-o": None,
    "W610208-o": "Did he or she have a stiff neck during the illness that led to death?",
    "W610298-o": "Was the bleeding in between her menstrual periods excessive?",
    "W610342-o": "Was her delivery normal vaginal, without forceps or vacuum?",
    "W610343-o": "Was her delivery vaginal, with forceps or vacuum?",
    "W610344-o": "Was her delivery a Caesarean section?",
    "W610387-o": "Was the baby's delivery normal vaginal, without forceps or vacuum?",
    "W610388-o": "Was the baby's delivery vaginal, with forceps or vacuum?",
    "W610389-o": "Was the baby's delivery a Caesarean section?",
    "W610356-o": "Is the child's mother still alive?",
}

# The form's other respondent questions: field -> (wording, note). None keeps WHO's
# wording, or takes the note from WHO's hint; "" means no note.
EXTRA = {
    "Id10058": (None, None),
    "Id10487": (None, "COVID-19 symptoms include fever, difficulty breathing, cough, extreme "
                      "fatigue, and changes in sense of smell or taste."),
    "Id10488": (None, ""),
    "Id10080": (None, None),
    "Id10081": (None, None),
    "Id10088": (None, None),
    "Id10097": (None, None),
    "Id10351": (None, "The maximum for a newborn is 27 days. Less than 24 hours = 0 days; "
                      "1 week = 7 days."),
    "Id10352_units": (None, None),
    "Id10126": (None, None),
    "Id10482": (None, None),
    "Id10483": (None, None),
    "Id10484": ("What was the result of the recent test by a health professional for "
                "COVID-19?",
                "If there was more than one test, give the result of the most recent one."),
    "Id10485": (None, "This means feeling so tired that he or she found it hard to get out "
                      "of bed and do routine things like taking a shower or changing "
                      "clothes."),
    "Id10486": (None, None),
    "Id10307": (None, None),
    "Id10308": (None, None),
    "Id10339": (None, None),
    "Id10366": (None, "If a weight recorded on the child's health card is mentioned, use it "
                      "rather than a remembered weight. The weight may be given in "
                      "kilograms; 1 kilogram = 1,000 grams."),
    "Id10379_unit": (None, None),
    "Id10392": ("In relation to the baby's mother receiving any vaccinations since reaching "
                "adulthood including during this pregnancy, how many doses did they "
                "receive?", None),
    "Id10416": (None, None),
    "Id10432": (None, None),
    "Id10433": ("Where or from whom did the family seek care?", None),
    "Id10435": ("Did a health care worker tell the family the cause of death?", None),
    "Id10436": ("If a health care worker told the family the cause of death, what did they "
                "say?", None),
    "Id10445": (None, None),
    "Id10446": (None, None),
    "age_group": (None, "Newborn (neonate): 0–27 completed days; child: 28 days through 11 "
                        "years; adult: 12 years and over. WHO asks this when the date of "
                        "birth or of death is unknown; CODA asks it when the case profile "
                        "has no age."),
}
COUNT_UNITS = {"Id10366": "grams", "Id10392": "doses", "Id10416": "times a day"}
# The form's age rows, asked as one question when the case profile has no age
AGE_ROWS = ("age_neonate_days", "age_child_unit", "age_child_days", "age_child_months",
            "age_child_years", "age_adult")
AGE_TEXT = "How old was the deceased?"
AGE_NOTE = ("Only asked when the case profile has no age. Less than 24 hours = 0 days. WHO "
            "requires an answer here and asks for a best estimate when the exact age is "
            "unknown; CODA's rule (the most exact answer the narrative allows, a range if "
            "need be, or unknown) applies instead.")

# Rows not asked: presets, the respondent and interviewer, consent, names, dates, extra
# demographics, date of marriage, civil registration, documents, the facility's name,
# and the narrative
LEFT_OUT = {
    "Id10002", "Id10003", "Id10007", "Id10007a", "Id10008", "Id10009", "Id10010",
    "Id10010a", "Id10010b", "Id10010c", "Id10013", "Id10017", "Id10018", "Id10061",
    "Id10062", "Id10020", "Id10021", "Id10022", "Id10023_a", "Id10023_b", "Id10024",
    "Id10051", "Id10052", "Id10053", "Id10054", "Id10055", "Id10057", "Id10063", "Id10064",
    "Id10065", "Id10066", "Id10060_check", "Id10060", "Id10069", "Id10069_a", "Id10070",
    "Id10071_check", "Id10071", "Id10072", "Id10073", "Id10429", "Id10430", "Id10431",
    "Id10437", "Id10438", "Id10439_check", "Id10439", "Id10440_check", "Id10440",
    "Id10441_check", "Id10441", "Id10442", "Id10443", "Id10444", "Id10462", "Id10463",
    "Id10464", "Id10465", "Id10466", "Id10467", "Id10468", "Id10469", "Id10470", "Id10471",
    "Id10472", "Id10473", "Id10434", "Id10476", "Id10477", "Id10478", "Id10479", "comment",
    "Id10008_check",
}
# Go-back checks: check row -> the question its message says to go back to, if any
CHECKS = {"id10260_check": None, "id1036X_check": None, "id10389_check": None,
          "id10414_check": None, "Id10310_check": "Id10296"}
# WHO's own message for this check addresses the interviewer
CHECK_MESSAGES = {"Id10310_check": "The answer to the not-pregnant confirmation says she "
                                   "was pregnant or had recently been pregnant, which the "
                                   "earlier answers don't show."}
# Gates WHO also opens on "doesn't know"; only "yes" opens them here
YES_ONLY = ("Id10077", "Id10104", "Id10370")
# The belly-pain duration is asked through the Id10196 unit row
SPOKEN_ROWS = {"W610197": ["id10196_unit"]}
# Hint sentences that don't carry over into notes
DROP_HINT = re.compile(r"(\"99|\"88|99\.\"|88\.\"|don't know|refuse|go back|"
                       r"should be asked only|read the question|respondent|^important is)",
                       re.I)


class XLSForm:
    """The WHO survey with its question skip logic and answer choice lists."""

    def __init__(self, path: str):
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
        self.rows, groups = {}, []
        for i, r in sheets["survey"].iterrows():
            kind, name = r["type"].strip(), r["name"].strip()
            if kind.startswith("begin"):
                groups.append((name, r["relevant"].strip()))
            elif kind.startswith("end") and kind != "end":
                if groups:
                    groups.pop()
            elif name:
                self.rows[name] = {
                    "type": kind, "row": i + 2,
                    "label": r["label::English"].strip(),
                    "hint": r["hint::English"].strip(),
                    "relevant": r["relevant"].strip(),
                    "groups": [rel for group, rel in groups if rel and group != "consented"],
                }
        self.lists = defaultdict(list)
        for _, r in sheets["choices"].iterrows():
            self.lists[r["list_name"]].append((r["name"].strip(), r["label::English"].strip()))

    def spoken(self, field: str) -> bool:
        """Whether the interviewer reads the row out loud (not a calculation, note or
        data-entry row)."""
        r = self.rows[field]
        if r["type"].startswith(("calculate", "note", "start", "end", "today")):
            return False
        return not re.match(r"^\(\w+\)\s*\[.*\]\s*[:?]?\s*$", r["label"])

    def fields(self, item: str) -> List[str]:
        """The spoken rows behind a probbase item (W610148-a -> Id10148_a, ...)."""
        if item[:7] in SPOKEN_ROWS:
            return SPOKEN_ROWS[item[:7]]
        stem = ("Id1" + item[3:7]).lower()
        return [f for f in self.rows if self.spoken(f)
                and (f.lower() == stem or f.lower().startswith(stem + "_"))]

    def relevant(self, field: str) -> str:
        """The row's skip logic and its groups' logic joined into one expression."""
        r = self.rows[field]
        parts = r["groups"] + ([r["relevant"]] if r["relevant"] else [])
        return " and ".join(f"({p})" for p in parts)

    def options(self, field: str) -> List[List[str]]:
        kind = self.rows[field]["type"].split()
        options = self.lists.get(kind[1], []) if len(kind) > 1 else []
        return [[name, label] for name, label in options if name not in SKIP_CHOICES
                and label.lower() not in ("don’t know", "doesn't know")]


def wording(label: str) -> str:
    """Apply a few wording changes in the form."""
    text = re.sub(r"^\(\w+\)\s*", "", label).strip().strip("[]").strip()
    for old, new in ((" (use local term)", ""), ("baby/ child", "baby/child"),
                     ("you/the baby's mother", "the baby's mother"),
                     ("you/the mother", "the baby's mother"),
                     ("her/his", "his or her"), ("his/her", "his or her"),
                     ("(s)he", "he or she"), ("s(h)e", "he or she"), ("(S)he", "He or she"),
                     ("he/she", "he or she"), ("s/he", "he or she")):
        text = text.replace(old, new)
    text = re.sub(r"\s+([?:,.])", r"\1", re.sub(r"\s+", " ", text))
    return text.rstrip(":").strip()


def note_from_hint(form: XLSForm, field: str) -> str:
    """A note that hints at how to answer."""
    sentences = re.split(r"(?<=[.!?])\s+", form.rows[field]["hint"])
    keep = []
    for s in (s.strip() for s in sentences):
        if not s or DROP_HINT.search(s):
            continue
        s = s.replace("Enter 1 unit only:", "Use one unit:").replace(
            "The answer could be given in another unit, but for data entry use", "Answer in")
        s = re.sub(r"\benter\b", "answer", s, flags=re.I).replace("record ", "answer ")
        s = s.replace("“0”", "0").replace('"0"', "0")
        s = s.replace("(Less than", "Less than").rstrip(")")
        keep.append(s[0].upper() + s[1:])
    return wording(" ".join(keep)) if keep else ""


def _implied_unit(label: str, default: str) -> str:
    return next((u for u in UNITS if u in label.lower()), default)


def _with_needs(rule: dict) -> dict:
    if rule["item"] in NEEDS:
        rule["needs"] = NEEDS[rule["item"]]
    return rule


def _number_question(form: XLSForm, stem: str) -> dict:
    base, rules = NUMBERS[stem]
    fields = list(form.fields(stem + "-a") or form.fields(stem + "-o"))
    unit_rows = [f for f in fields if form.rows[f]["type"].startswith("select_one")]
    if unit_rows:
        ask = unit_rows[0]
        asked = [f for f in fields if f in unit_rows or ask not in form.rows[f]["relevant"]]
        units = [name.lower() for name, _ in form.options(ask)]
        fields += [f for f, r in form.rows.items()
                   if "${" + ask + "}" in r["relevant"] and f not in fields]
    else:
        ask, asked = fields[-1], fields
        units = [_implied_unit(form.rows[ask]["label"], base)]
    converts = []
    for suffix, rule in rules.items():
        convert = {"item": f"{stem}-{suffix}", "op": rule[0]}
        if rule[0] == "range":
            convert["range"] = [rule[1], rule[2]]
        else:
            convert["value"] = rule[1]
        converts.append(_with_needs(convert))
    q = {"id": "Id1" + stem[3:7], "kind": "number",
         "text": QUESTION_WORDING.get(stem, wording(form.rows[ask]["label"])),
         "fields": fields, "relevant": [form.relevant(f) for f in asked], "units": units,
         "base_unit": base, "note": note_from_hint(form, ask), "converts": converts}
    # WHO asks newborns for durations in days
    if base in ("minutes", "hours", "days", "weeks", "months"):
        newborn = [f for f in asked if "isNeonatal" in form.rows[f]["relevant"]
                   and form.rows[f]["type"].startswith("integer")]
        if newborn and newborn[0] != ask:
            q["text_newborn"] = wording(form.rows[newborn[0]]["label"])
            q["note_newborn"] = note_from_hint(form, newborn[0])
    return q


def _choice_questions(form: XLSForm, stem: str) -> List[dict]:
    multi = stem in MULTI
    field, rules = MULTI[stem] if multi else CHOICES[stem]
    options = form.options(field)
    names = {name for name, _ in options}
    converts = []
    for suffix, rule in rules.items():
        convert = {"item": f"{stem}-{suffix}"}
        if multi:
            convert.update(yes_if=rule[1], mode=rule[0])
            if len(rule) > 2:
                convert["yes_if_all"] = rule[2]
        else:
            convert["yes_if"] = rule
        unknown = set(convert["yes_if"] + convert.get("yes_if_all", [])) - names
        if unknown:
            raise ValueError(f"{stem}-{suffix}: unknown choices {sorted(unknown)}")
        converts.append(_with_needs(convert))
    questions = [{"id": field if stem in ADULT_VERSION else "Id1" + stem[3:7],
                  "kind": "multi" if multi else "choice",
                  "text": QUESTION_WORDING.get(stem, wording(form.rows[field]["label"])),
                  "fields": [field], "relevant": [form.relevant(field)], "choices": options,
                  "converts": converts}]
    if stem in ADULT_VERSION:
        adult, text = ADULT_VERSION[stem]
        questions.append({"id": adult, "kind": "yesno", "text": text, "fields": [adult],
                          "relevant": [form.relevant(adult)],
                          "converts": [{"item": f"{stem}-a"}]})
    return questions


def _yesno_question(form: XLSForm, item: str, probbase_question) -> dict:
    field = form.fields(item)[0]
    text = YESNO_WORDING.get(item, wording(form.rows[field]["label"]))
    return {"id": field, "kind": "yesno", "text": text or probbase_question.text,
            "fields": [field], "relevant": [form.relevant(field)],
            "note": probbase_question.note, "converts": [_with_needs({"item": item})]}


def _extra_question(form: XLSForm, field: str) -> dict:
    text, note = EXTRA[field]
    row = form.rows[field]
    kind = row["type"].split()
    q = {"id": field.split("_unit")[0], "text": text or wording(row["label"]),
         "fields": [field], "relevant": [form.relevant(field)] if field != "age_group" else [],
         "note": note_from_hint(form, field) if note is None else note, "converts": []}
    if kind[0] == "select_one" and field.endswith(("_units", "_unit")):
        q.update(kind="number", units=[name.lower() for name, _ in form.options(field)])
        q["fields"] += [f for f, r in form.rows.items() if "${" + field + "}" in r["relevant"]]
    elif kind[0] == "integer":
        unit = COUNT_UNITS.get(field, _implied_unit(row["label"], "number"))
        q.update(kind="number", units=[unit], base_unit=unit)
    elif kind[0] == "text":
        q["kind"] = "text"
    elif kind[0] == "select_multiple":
        q.update(kind="multi", choices=form.options(field))
    elif kind[1].startswith("YES_NO"):
        q["kind"] = "yesno"
    else:
        q.update(kind="choice", choices=form.options(field))
    return q


def _checks(form: XLSForm, questions: List[dict]) -> List[dict]:
    """The go-back checks, along with the questions it says to re-ask."""
    owner = {f: q["id"] for q in questions for f in q["fields"]}
    first_row = {q["id"]: min(form.rows[f]["row"] for f in q["fields"]) for q in questions}
    checks = []
    for name, back_to in CHECKS.items():
        row = form.rows[name]
        if back_to:
            start = form.rows[back_to]["row"]
            reask = [q["id"] for q in questions if start <= first_row[q["id"]] < row["row"]]
        else:
            reask = sorted({owner[f] for f in refs(parse(row["relevant"]))},
                           key=first_row.get)
        message = CHECK_MESSAGES.get(name) or wording(row["label"])
        checks.append({"id": name, "message": message, "condition": row["relevant"],
                       "reask": reask})
    return checks


def _validate(form: XLSForm, questions: List[dict]) -> None:
    """Check that ids are unique, every spoken row is asked or left out exactly once, and
    every expression parses cleanly."""
    ids = Counter(q["id"] for q in questions)
    rows = Counter(f for q in questions for f in q["fields"])
    duplicates = sorted(k for counts in (ids, rows) for k, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate ids or rows: {duplicates}")
    missing = [f for f in form.rows if form.spoken(f) and f not in rows and f not in LEFT_OUT]
    if missing:
        raise ValueError(f"spoken rows neither asked nor left out: {missing}")
    for q in questions:
        for expr in q["relevant"]:
            parse(expr)


def build(path: str) -> dict:
    """Build the bank from the WHO form's XLSForm."""
    form = XLSForm(path)
    probbase = {q.id[len(PREFIX):]: q for q in load_question_bank()}
    questions = [_number_question(form, stem) for stem in NUMBERS]
    for stem in list(CHOICES) + list(MULTI):
        questions += _choice_questions(form, stem)
    asked_as = set(NUMBERS) | set(CHOICES) | set(MULTI)
    questions += [_yesno_question(form, item, q) for item, q in probbase.items()
                  if item[:7] not in asked_as and not item.startswith("W610022")]
    questions += [_extra_question(form, field) for field in EXTRA]
    questions.append({"id": "age", "kind": "number", "text": AGE_TEXT,
                      "fields": list(AGE_ROWS), "relevant": [],
                      "units": ["days", "months", "years"], "base_unit": "days",
                      "note": AGE_NOTE, "converts": []})
    for q in questions:
        if "" in q["relevant"]:
            q["relevant"] = []
    questions.sort(key=lambda q: min(form.rows[f]["row"] for f in q["fields"]))
    _validate(form, questions)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    keep = ("relevant", "converts")
    return {"source": {"form": "WHO 2016 VA questionnaire, ODK XLSForm v1.5.3",
                       "file": path.split("/")[-1], "sha256": digest},
            "checks": _checks(form, questions), "yes_only": list(YES_ONLY),
            "questions": [{k: v for k, v in q.items() if v not in ("", [], None) or k in keep}
                          for q in questions]}


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python -m coda.questionnaire.who_bank "
                 "<WHOVA2016_v1_5_3_XLS_form_for_ODK.xlsx> [output.json]")
    out = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_WHO_BANK
    bank = build(sys.argv[1])
    with open(out, "w") as fh:
        json.dump(bank, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    kinds = Counter(q["kind"] for q in bank["questions"])
    items = sum(len(q["converts"]) for q in bank["questions"])
    print(f"{len(bank['questions'])} questions ({dict(kinds)}), "
          f"{items} probbase items -> {out}")


if __name__ == "__main__":
    main()
