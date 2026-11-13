"""Backward Think-on-Graph walks from a candidate cause's ICD-10 code.

An LLM walks the ICD-10 hierarchy and ACME causal edges one hop at a time,
keeping the facts relevant to the case, and the cause's typical ProbBase
findings are checked against the case. Both take an async
``call(step, system_prompt, user_prompt, schema, **context)`` that makes one
structured LLM call and raises LLMCallFailed if it fails.
"""

import logging
import re
from pathlib import Path

from coda.inference.champs_prompted.base_agent import LLMCallFailed
from coda.kg.graph import GRADES, KGView

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_HOPS = 3
MAX_CHOICES = 3  # Facts the LLM may pick per hop
MAX_SHOWN = 30  # Facts shown per node, relation and direction at each hop
AGE_QUESTION_PREFIX = "who.va.q:W610022-"  # ProbBase age-bracket questions to skip
PREGNANCY_QUESTION = "who.va.q:W610310-o"  # "NEITHER pregnant...", to skip for children


def read_prompt(filename):
    """Read a prompt text file of the ToG walks."""
    return (PROMPTS_DIR / filename).read_text().strip()


WALK_STEP_PROMPT = read_prompt("walk_step.txt") \
    .format(max_choices=MAX_CHOICES)
FINDING_CHECK_PROMPT = read_prompt("finding_check.txt")

WALK_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen": {
            "type": "array", "items": {"type": "integer"},
            "maxItems": MAX_CHOICES,
            "description": "Numbers of the facts to follow; empty to stop.",
        },
        "reason": {"type": "string"},
    },
    "required": ["chosen", "reason"],
    "additionalProperties": False,
}

FINDING_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "status": {"type": "string", "enum": [
                        "yes", "no", "not_mentioned"]},
                    "quote": {"type": "string"},
                },
                "required": ["number", "status", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def format_triple(graph: KGView, head, relation, tail):
    return f"({graph.name(head)}, {relation}, {graph.name(tail)})"


def step_options(graph: KGView, frontier, visited):
    """Return the ((head, relation, tail), next_node) facts one hop away. Also
    return how many unvisited neighbors were cut by MAX_SHOWN.
    """
    options, omitted = {}, 0
    for node in frontier:
        for relation, direction in graph.relations(node):
            neighbors = [n for n in graph.neighbors(node, relation, direction)
                         if n not in visited]
            omitted += max(0, len(neighbors) - MAX_SHOWN)
            for neighbor in neighbors[:MAX_SHOWN]:
                fact = (node, relation, neighbor) if direction == "out" \
                    else (neighbor, relation, node)
                options.setdefault(fact, neighbor)
    return list(options.items()), omitted


async def walk_back(graph: KGView, call, case_input, cause, code):
    """Walk the KG from a candidate's ICD-10 code; return the kept facts. Stops
    after MAX_HOPS hops, when the LLM picks no fact, or when a call fails.
    """
    start = graph.resolve(code)
    if start is None:
        return []
    frontier, visited, triples = [start], {start}, []
    header = graph.name(start)
    if graph.same_as(start):
        header += f"; same as: {graph.same_as(start)}"
    for hop in range(1, MAX_HOPS + 1):
        options, omitted = step_options(graph, frontier, visited)
        if not options:
            break
        facts = [format_triple(graph, *fact) for fact, _ in options]
        user_prompt = (
            f"{case_input}\n\n"
            f"## CANDIDATE CAUSE\n{cause} ({header})\n\n"
            "## FACTS SO FAR\n" + ("\n".join(triples) or "(none)") + "\n\n"
            "## NEXT FACTS\n"
            + "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1)))
        if omitted:
            user_prompt += f"\n({omitted} more facts not shown)"
        try:
            response = await call("walk_step", WALK_STEP_PROMPT, user_prompt,
                                  WALK_STEP_SCHEMA, candidate=cause, hop=hop,
                                  options=facts)
        except LLMCallFailed:
            logger.warning("Walk from %s stopped at hop %d", code, hop)
            break
        picks = [i for i in dict.fromkeys(response.get("chosen", []))
                 if 1 <= i <= len(options)][:MAX_CHOICES]
        if not picks:
            break
        triples += [facts[i - 1] for i in picks]
        frontier = list(dict.fromkeys(options[i - 1][1] for i in picks))
        visited.update(frontier)
    return triples


def typical_findings(graph: KGView, code):
    """Return {question: (grade, WHO VA cause)} for a code's strong ProbBase
    findings. A finding shared by several WHO VA causes of the code keeps its
    best grade.
    """
    links = sorted(((q, g, who_va) for who_va in graph.who_va_causes(code)
                    for q, g in graph.strong_findings(who_va)
                    if not q.startswith(AGE_QUESTION_PREFIX)
                    and q != PREGNANCY_QUESTION),
                   key=lambda link: GRADES.index(link[1]), reverse=True)
    return {q: (g, who_va) for q, g, who_va in links}


def normalized(text):
    return " ".join(text.lower().split())


def checked_status(item, case_text):
    """Return a finding's answer, or not_mentioned unless every part of its
    quote, split at an ellipsis, is from the case. A part may start with one
    added word, such as "she" in "she vomited" for "...and vomited".
    """
    def found(part):
        rest = part.partition(" ")[2]
        return part in case_text or (rest and rest in case_text)

    parts = [normalized(part).strip(" \"'.,")
             for part in re.split(r"\.\.\.|…", item.get("quote", ""))]
    parts = [part for part in parts if part]
    return item["status"] if parts and all(map(found, parts)) \
        else "not_mentioned"


async def check_findings(graph: KGView, call, case_input, codes):
    """Check the candidates' typical ProbBase findings against the case in one
    call, so that each question gets one answer. An answer counts only if it
    quotes the case.

    Return a list per code of dicts with each finding, its grade, the answer the
    grade is for, its WHO VA cause and the case's answer as status, or None for
    every code if the call failed.
    """
    findings = [typical_findings(graph, code) for code in codes]
    questions = sorted({q for typical in findings for q in typical})
    if not questions:
        return [[] for _ in codes]
    user_prompt = (
        f"{case_input}\n\n## FINDINGS\n"
        + "\n".join(f"{i}. {graph.name(question)}"
                    for i, question in enumerate(questions, 1)))
    try:
        response = await call("finding_check", FINDING_CHECK_PROMPT,
                              user_prompt, FINDING_CHECK_SCHEMA)
    except LLMCallFailed:
        logger.warning("Finding check for %s failed", codes)
        return [None for _ in codes]
    case_text = normalized(case_input)
    statuses = {item["number"]: checked_status(item, case_text)
                for item in response.get("findings", [])}
    numbers = {question: i for i, question in enumerate(questions, 1)}
    return [[{"finding": graph.name(question), "grade": grade,
              "graded_answer": graph.graded_answer(question),
              "who_va_cause": graph.name(who_va),
              "status": statuses.get(numbers[question], "not_mentioned")}
             for question, (grade, who_va) in sorted(typical.items())]
            for typical in findings]


def shown_findings(findings):
    """Return the typical findings the case answers the way their grade
    describes.
    """
    return [f for f in findings or [] if f["status"] == f["graded_answer"]]


def format_evidence(candidate, facts=True):
    """Render a candidate's walk facts, unless facts is False, and the typical
    findings the case shows, which are those answered the way their grade
    describes. As in InterVA, the other answers are not evidence, so they are
    only counted.
    """
    lines = [f"### {candidate['cause']} (ICD-10 {candidate['code']}, "
             f"first-opinion probability {candidate['score']:.2f})"]
    if facts:
        if candidate["same_as"]:
            lines.append(f"Same as: {candidate['same_as']}")
        lines.append("Walk facts:")
        lines += [f"- {triple}" for triple in candidate["triples"]] \
            or ["- (none)"]
    lines.append("Typical findings the case shows:")
    findings = candidate["findings"]
    if findings is None:
        lines.append("- (finding check failed)")
    elif not findings:
        lines.append("- (none in the KG)")
    else:
        shown = shown_findings(findings)
        lines += [f"- [{f['status']}] {f['finding']} (grade {f['grade']} "
                  f"for {f['who_va_cause']} on answer {f['graded_answer']})"
                  for f in shown]
        lines.append(f"- ({len(findings) - len(shown)} other typical findings "
                     "were answered otherwise or not mentioned)")
    return "\n".join(lines)
