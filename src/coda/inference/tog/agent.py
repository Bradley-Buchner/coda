"""Prompted ICD-10 agent that re-ranks its causes with backward Think-on-Graph
(ToG) walks.

The current prototype reads the KG from the exported TSV files and makes up to 14
LLM calls per inference.
"""

import logging
from functools import partial
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
    format_evidence,
    read_prompt,
    walk_back,
)
from coda.kg.graph import KGView

logger = logging.getLogger(__name__)

MAX_CITED = 5  # Facts, and separately findings, the final ranking may cite
MAX_CAUSES = 3  # Causes in a ranking, as in the first opinion's schema
RANKING_PROMPT = read_prompt("final_ranking.txt").format(
    allowed_causes=", ".join(CHAMPS_GROUP_CAUSES), max_cited=MAX_CITED)
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
    from its ICD-10 code and its typical findings (in the form of ProbBase
    items) are checked against the case; a final call ranks the causes with
    that evidence. The result adds a kg_evidence entry, and keeps the first
    opinion if the ranking fails. The calls of the last inference are kept
    in self.trace.
    """

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

        case_input = self.build_user_prompt()
        call = partial(self.call_step, chunk_id)
        candidates = []
        for cause in first_opinion["causes"].values():
            name = cause["name"]
            code = cause["identifiers"][self.coding_system]
            candidates.append({
                "cause": name, "code": code, "score": cause["score"],
                "same_as": self.graph.same_as(code),
                "triples": await walk_back(self.graph, call, case_input,
                                           name, code),
                "findings": await check_findings(self.graph, call,
                                                 case_input, name, code),
            })
        evidence = {"candidates": candidates, "cited": []}
        if not candidates:
            return {**first_opinion, "kg_evidence": evidence}

        user_prompt = (f"{case_input}\n\n## CANDIDATES\n"
                       + "\n\n".join(format_evidence(c) for c in candidates))
        try:
            response = await call("final_ranking", RANKING_PROMPT,
                                  user_prompt, RANKING_SCHEMA)
        except LLMCallFailed:
            return {**first_opinion, "kg_evidence": evidence}
        ranked = self._parse_response(response)
        if not ranked["causes"]:
            logger.warning("Final ranking named no allowed cause; keeping "
                           "the first opinion")
            return {**first_opinion, "kg_evidence": evidence}
        # Keep only citations of the evidence, not of the case, as the
        # longest fact or finding each one contains
        items = {item for c in candidates for item in c["triples"]
                 + [f["finding"] for f in c["findings"] or []]}
        cited = []
        for text in (response.get("cited_facts", [])
                     + response.get("cited_findings", [])):
            found = [item for item in items if item in text]
            if found:
                cited.append(max(found, key=len))
        evidence["cited"] = list(dict.fromkeys(cited))
        updates = {self._match_cause(update["cause_name"]): update
                   for update in response.get("candidate_updates", [])}
        causes = apply_updates(first_opinion["causes"], ranked["causes"],
                               updates)
        reasoning = " ".join([ranked["reasoning"], *explain_changes(
            first_opinion["causes"], causes, updates)])
        return {**first_opinion, "causes": causes, "reasoning": reasoning,
                "kg_evidence": evidence}
