"""Prompted ICD-10 agents that re-rank their causes with KG evidence: backward
Think-on-Graph (ToG) walks and typical findings, the findings alone, or
reasoning steps over numbered ACME rules and findings, with or without walks.

The current prototype reads the KG from the exported TSV files. The full agent
makes up to 12 LLM calls per inference, the findings-only and reasoning agents
3.
"""

import logging
from functools import cache, partial
from typing import List

from gilda import Annotation

from coda.inference.champs_prompted.base_agent import (
    LLMCallFailed,
    normalize_scores,
)
from coda.inference.champs_prompted.icd10_agent import (
    CHAMPS_GROUP_CAUSES,
    ChampsPromptedIcd10Agent,
    build_cod_output_schema,
)
from coda.inference.tog.backward import (
    check_findings,
    checked_status,
    format_evidence,
    normalized,
    read_prompt,
    walk_back,
)
from coda.inference.tog.reasoning import (
    ROLES,
    check_step,
    kg_items,
    render_items,
)
from coda.kg.check import check_top_causes
from coda.kg.graph import KGView
from coda.kg.query import KGQuery

logger = logging.getLogger(__name__)

MAX_CITED = 5  # Facts, and separately findings, the final ranking may cite
MAX_CAUSES = 3  # Causes in a ranking, as in the first opinion's schema
MAX_STEPS = 8  # Reasoning steps of the reasoning agents
RANKING_PROMPT, FINDINGS_RANKING_PROMPT, REASONING_PROMPT = (
    read_prompt(name).format(allowed_causes=", ".join(CHAMPS_GROUP_CAUSES),
                             max_cited=MAX_CITED)
    for name in ("final_ranking.txt", "findings_ranking.txt",
                 "kg_reasoning.txt"))
default_kg = cache(KGQuery)
COD_PROPERTIES = build_cod_output_schema()["properties"]
CITED_SCHEMA = {"type": "array", "items": {"type": "string"},
                "maxItems": MAX_CITED}
RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        # First, so that each direction is decided before the probabilities
        "candidate_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cause_name": {"type": "string"},
                    "direction": {"type": "string",
                                  "enum": ["up", "down", "no_change"]},
                    "evidence": {"type": "string"},
                },
                "required": ["cause_name", "direction", "evidence"],
                "additionalProperties": False,
            },
        },
        "reasoning": COD_PROPERTIES["reasoning"],
        "top_causes": COD_PROPERTIES["top_causes"],
        "cited_facts": CITED_SCHEMA,
        "cited_findings": CITED_SCHEMA,
    },
    "required": ["candidate_updates", "reasoning", "top_causes", "cited_facts",
                 "cited_findings"],
    "additionalProperties": False,
}
FINDINGS_RANKING_SCHEMA = {
    **RANKING_SCHEMA,
    "properties": {key: value for key, value in RANKING_SCHEMA["properties"]
                   .items() if key != "cited_facts"},
    "required": [key for key in RANKING_SCHEMA["required"]
                 if key != "cited_facts"],
}
NAMES_SCHEMA = {"type": "array", "items": {"type": "string"}}
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "role": {"type": "string", "enum": ROLES},
        "favors": NAMES_SCHEMA,
        "disfavors": NAMES_SCHEMA,
        "underlying": {"type": "string", "description": (
            "The cause this step concludes is the underlying cause of this "
            "death, or empty.")},
        "basis": {"type": "array", "minItems": 1, "items": {
            "type": "string", "pattern": "^([RFW][0-9]+|narrative)$"}},
        "quote": {"type": "string"},
    },
    "required": ["claim", "role", "favors", "disfavors", "underlying",
                 "basis", "quote"],
    "additionalProperties": False,
}
REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_steps": {"type": "array", "items": STEP_SCHEMA,
                            "minItems": 1, "maxItems": MAX_STEPS},
        "candidate_updates": RANKING_SCHEMA["properties"]["candidate_updates"],
        "top_causes": COD_PROPERTIES["top_causes"],
    },
    "required": ["reasoning_steps", "candidate_updates", "top_causes"],
    "additionalProperties": False,
}


def apply_updates(first_causes, final_causes, updates):
    """Return the final causes, with no_change causes at their first-opinion
    probabilities and the others rescaled to fill what is left.
    """
    kept = {key: {**cause} for key, cause in first_causes.items()
            if updates.get(cause["name"], {}).get("direction") == "no_change"}
    moved = {key: cause for key, cause in final_causes.items()
             if key not in kept}
    total = sum(cause["score"] for cause in moved.values())
    if total > 0:
        room = 1 - sum(cause["score"] for cause in kept.values())
        for cause in moved.values():
            cause["score"] = cause["score"] * room / total
    causes = sorted({**kept, **moved}.items(),
                    key=lambda item: item[1]["score"], reverse=True)
    return normalize_scores(dict(causes[:MAX_CAUSES]))


def explain_changes(first_causes, final_causes, updates):
    """Return a sentence per cause whose probability differs from the first
    opinion, flagging a stated direction that the numbers contradict.
    """
    sentences = []
    for key in dict.fromkeys([*final_causes, *first_causes]):
        name = (final_causes.get(key) or first_causes[key])["name"]
        old = first_causes.get(key, {}).get("score", 0)
        new = final_causes.get(key, {}).get("score", 0)
        if f"{old:.2f}" == f"{new:.2f}":
            continue
        update = updates.get(name, {})
        evidence = update.get("evidence", "no reason given").rstrip(".")
        sentence = f"{name} moved from {old:.2f} to {new:.2f}: {evidence}."
        if "direction" in update \
                and update["direction"] != ("up" if new > old else "down"):
            sentence += f" (Stated direction: {update['direction']}.)"
        sentences.append(sentence)
    return sentences


class ToGBackwardIcd10Agent(ChampsPromptedIcd10Agent):
    """Prompted ICD-10 agent re-ranking its causes with KG evidence.

    For each cause of the first opinion, a backward walk collects KG facts
    from its ICD-10 code; one call then checks all the causes' typical findings
    (in the form of ProbBase items) against the case, and a final call ranks
    the causes with that evidence. The result adds a kg_evidence entry, and keeps the first
    opinion if the ranking fails. The calls of the last inference are kept
    in self.trace.
    """

    walks = True
    ranking_prompt = RANKING_PROMPT
    ranking_schema = RANKING_SCHEMA

    def __init__(self, llm_client, graph: KGView, **kwargs):
        super().__init__(llm_client, **kwargs)
        self.graph = graph
        self.trace = []

    def create_session_agent(self) -> "ToGBackwardIcd10Agent":
        return self.__class__(
            llm_client=self.llm_client,
            graph=self.graph,
            use_diagnosis_standard=self.use_diagnosis_standard,
            llm_semaphore=self.llm_semaphore,
            num_questions=self.num_questions,
            question_style=self.question_style,
        )

    async def call_step(self, chunk_id, step, system_prompt, user_prompt,
                        schema, **context):
        """Make one structured LLM call and record it in the trace."""
        entry = {"step": step, **context, "system_prompt": system_prompt,
                 "user_prompt": user_prompt, "response": None}
        self.trace.append(entry)
        entry["response"] = await self.call_llm(
            chunk_id, self.llm_client.call_with_schema,
            system_prompt=system_prompt, user_prompt=user_prompt,
            schema=schema, schema_name=step, temperature=0)
        return entry["response"]

    async def infer(self, chunk_id: str, text: str,
                    annotations: List[Annotation]):
        self.trace = []
        first_opinion = await super().infer(chunk_id, text, annotations)
        self.trace.append({"step": "first_opinion",
                           "response": first_opinion})
        return await self.rerank(chunk_id, first_opinion,
                                 self.build_user_prompt())

    async def rerank(self, chunk_id, first_opinion, case_input):
        """Collect the first opinion's evidence and rank its causes with it."""
        call = partial(self.call_step, chunk_id)
        candidates = []
        for cause in first_opinion["causes"].values():
            name = cause["name"]
            code = cause["identifiers"][self.coding_system]
            candidates.append({
                "cause": name, "code": code, "score": cause["score"],
                "same_as": self.graph.same_as(code) if self.walks else "",
                "triples": await walk_back(self.graph, call, case_input,
                                           name, code) if self.walks else [],
            })
        checked = await check_findings(self.graph, call, case_input,
                                       [c["code"] for c in candidates])
        for candidate, findings in zip(candidates, checked):
            candidate["findings"] = findings
        if not candidates:
            return {**first_opinion,
                    "kg_evidence": {"candidates": [], "cited": []}}
        return await self.rank(chunk_id, first_opinion, case_input, candidates)

    def evidence_block(self, first_opinion, candidates, evidence):
        """Return the part of the final ranking prompt after the case."""
        return "## CANDIDATES\n" + "\n\n".join(
            format_evidence(c, self.walks) for c in candidates)

    def read_evidence(self, response, ranked, evidence, case_input):
        """Record in evidence what the final ranking cited; return its
        reasoning.
        """
        # Keep only citations of the evidence, not of the case, as the
        # longest fact or finding each one contains
        candidates = evidence["candidates"]
        items = {item for c in candidates for item in c["triples"]
                 + [f["finding"] for f in c["findings"] or []]}
        cited = []
        for text in (response.get("cited_facts", [])
                     + response.get("cited_findings", [])):
            found = [item for item in items if item in text]
            if found:
                cited.append(max(found, key=len))
        evidence["cited"] = list(dict.fromkeys(cited))
        return ranked["reasoning"]

    async def rank(self, chunk_id, first_opinion, case_input, candidates,
                   **kwargs):
        """Rank the causes with the candidates' evidence, keeping the first
        opinion if the ranking fails. Keyword arguments go to evidence_block.
        """
        evidence = {"candidates": candidates, "cited": []}
        user_prompt = f"{case_input}\n\n" + self.evidence_block(
            first_opinion, candidates, evidence, **kwargs)
        try:
            response = await self.call_step(chunk_id, "final_ranking",
                                            self.ranking_prompt, user_prompt,
                                            self.ranking_schema)
        except LLMCallFailed:
            return {**first_opinion, "kg_evidence": evidence}
        ranked = self._parse_response(response)
        if not ranked["causes"]:
            logger.warning("Final ranking named no allowed cause; keeping "
                           "the first opinion")
            return {**first_opinion, "kg_evidence": evidence}
        reasoning = self.read_evidence(response, ranked, evidence, case_input)
        updates = {self._match_cause(update["cause_name"]): update
                   for update in response.get("candidate_updates", [])}
        causes = apply_updates(first_opinion["causes"], ranked["causes"],
                               updates)
        reasoning = " ".join([reasoning, *explain_changes(
            first_opinion["causes"], causes, updates)])
        return {**first_opinion, "causes": causes, "reasoning": reasoning,
                "kg_evidence": evidence}


class ToGFindingsIcd10Agent(ToGBackwardIcd10Agent):
    """ToGBackwardIcd10Agent without the walks: the final call ranks the causes
    with their typical findings alone, after one call that checks them.
    """

    walks = False
    ranking_prompt = FINDINGS_RANKING_PROMPT
    ranking_schema = FINDINGS_RANKING_SCHEMA


class ToGReasoningIcd10Agent(ToGBackwardIcd10Agent):
    """ToGFindingsIcd10Agent whose final call reasons step by step over
    numbered KG items: ACME rules on how the candidates relate and the typical
    findings the case shows. Each step names the items it rests on, which are
    checked against it.
    """

    walks = False
    ranking_prompt = REASONING_PROMPT
    ranking_schema = REASONING_SCHEMA

    def __init__(self, llm_client, graph: KGView, kg: KGQuery = None,
                 **kwargs):
        super().__init__(llm_client, graph=graph, **kwargs)
        self.kg = kg or default_kg()

    def create_session_agent(self):
        agent = super().create_session_agent()
        agent.kg = self.kg
        return agent

    def evidence_block(self, first_opinion, candidates, evidence, items=None):
        """Return the candidates and the KG items, which are given, or else
        built from the first opinion and the candidates' evidence.
        """
        if items is None:
            causes = first_opinion["causes"]
            items = kg_items(check_top_causes(self.kg, causes), causes,
                             candidates, self.walks)
        evidence["items"] = items
        return "## CANDIDATES\n" + "\n".join(
            f"- {c['cause']} (ICD-10 {c['code']}, first-opinion probability "
            f"{c['score']:.2f})" for c in candidates) \
            + "\n\n" + render_items(items)

    def read_evidence(self, response, ranked, evidence, case_input):
        """Record each reasoning step with its causes matched, the cited ids
        that were and weren't shown, its check against the shown items and
        whether its quote is from the case. Return the steps' claims.
        """
        def matched(names):
            return [name for name in map(self._match_cause, names) if name]

        shown = {item["id"]: item for item in evidence["items"]}
        case_text = normalized(case_input)
        steps = []
        for step in response.get("reasoning_steps", []):
            step = {**step, "favors": matched(step["favors"]),
                    "disfavors": matched(step["disfavors"]),
                    "underlying": self._match_cause(step["underlying"]) or ""}
            step["valid"] = [i for i in step["basis"] if i in shown]
            step["invalid"] = [i for i in step["basis"]
                               if i not in shown and i != "narrative"]
            step["check"] = check_step(step, evidence["items"])
            step["quote_ok"] = checked_status(
                {"status": "yes", "quote": step["quote"]}, case_text) == "yes" \
                if "narrative" in step["basis"] else None
            steps.append(step)
        evidence["steps"] = steps
        evidence["cited"] = list(dict.fromkeys(
            shown[i]["text"] for step in steps for i in step["valid"]))
        return " ".join(step["claim"] for step in steps)


class ToGReasoningWalksIcd10Agent(ToGReasoningIcd10Agent):
    """ToGReasoningIcd10Agent that also walks the KG and shows the walk facts
    as W items.
    """

    walks = True
