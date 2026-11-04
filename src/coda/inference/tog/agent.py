"""Prompted ICD-10 agent that re-ranks its causes with backward Think-on-Graph
(ToG) walks.

The current prototype reads the KG from the exported TSV files and makes up to 14
LLM calls per inference.
"""

import logging
from functools import partial
from typing import List

from gilda import Annotation

from coda.inference.champs_prompted.base_agent import LLMCallFailed
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

MAX_CITED = 5  # Facts or findings the final ranking may cite
RANKING_PROMPT = read_prompt("final_ranking.txt").format(
    allowed_causes=", ".join(CHAMPS_GROUP_CAUSES), max_cited=MAX_CITED)
COD_PROPERTIES = build_cod_output_schema()["properties"]
RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": COD_PROPERTIES["reasoning"],
        "top_causes": COD_PROPERTIES["top_causes"],
        "cited": {"type": "array", "items": {"type": "string"},
                  "maxItems": MAX_CITED},
    },
    "required": ["reasoning", "top_causes", "cited"],
    "additionalProperties": False,
}


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
        evidence["cited"] = response.get("cited", [])
        return {**first_opinion, "causes": ranked["causes"],
                "reasoning": ranked["reasoning"], "kg_evidence": evidence}
