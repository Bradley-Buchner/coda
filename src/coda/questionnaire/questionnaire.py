"""The questionnaire step of the CHAMPS prompted agents. Answers standardized VA questions
from the dialogue before every inference."""

from typing import Any, Dict, List, Optional

from coda.config import PROMPTS
from coda.metadata import Profile
from coda.questionnaire.answerer import QuestionAnswer, QuestionAnswerer, format_answers
from coda.questionnaire.bank import UNKNOWN, Question, load_question_bank
from coda.questionnaire.retriever import QuestionRetriever
from coda.questionnaire.who_answerer import WhoAnswerer, shown, who_record
from coda.questionnaire.who_form import WhoForm, load_who_form

# Retrieval searches only the most recent chunks -> each new topic gets its own questions
RECENT_CHUNKS = 2

# Profile sexes as the WHO form names them
WHO_SEXES = {"female": "female", "male": "male", "ambiguous": "undetermined"}


class Questionnaire:
    """Answers the questionnaire with one of two strategies.

    who_form asks the WHO 2016 VA form's questions in rounds on the whole transcript and
    converts the answers to probbase items with `bank`. retrieval answers the bank
    questions most semantically similar to the latest dialogue and keeps the known answers.
    """

    def __init__(self, strategy: str, answerer, retriever: Optional[QuestionRetriever] = None,
                 form: Optional[WhoForm] = None, bank: Optional[List[Question]] = None):
        self.strategy = strategy
        self.answerer = answerer
        self.retriever = retriever
        self.form = form
        self.bank = bank

    def update(self, answers: Any, dialogue_history: list, profile: Optional[Profile]) -> Any:
        """Return the answers of an earlier update (None at first) updated from the
        dialogue -> the form's record for who_form, the known answers for retrieval."""
        transcript = " ".join(text for _, _, text, _ in dialogue_history)
        if not transcript.strip():
            return answers
        if self.strategy == "who_form":
            return self._update_who_form(answers, transcript, profile)
        recent = " ".join(text for _, _, text, _ in dialogue_history[-RECENT_CHUNKS:])
        return self._update_retrieval(answers, transcript, recent)

    def _update_who_form(self, record: Optional[Dict[str, Any]], transcript: str,
                         profile: Optional[Profile]) -> Dict[str, Any]:
        age_days = profile.age.days if profile and profile.age else None
        sex = WHO_SEXES.get(profile.sex) if profile else None
        stillbirth = bool(profile and profile.stillbirth)
        answers, interview, complete = self.answerer.answer_form(
            transcript, self.form, age_days, sex, stillbirth)
        if not complete and record is not None:
            return record
        return {**who_record(answers, interview, self.bank), "complete": complete}

    def _update_retrieval(self, answers: Optional[List[Dict[str, str]]], transcript: str,
                          recent: str) -> List[Dict[str, str]]:
        known = {a["id"]: a for a in answers or []}
        retrieved = self.retriever.retrieve(recent)
        # A later "unknown" never erases a known answer, whereas a later yes/no does
        for a in self.answerer.answer(transcript, [r.question for r in retrieved]):
            if a.answer != UNKNOWN:
                known[a.id] = a.to_dict()
        return list(known.values())

    def format(self, answers: Any, transcript: str) -> str:
        """Format the known answers as prompt lines, one line per quoted statement."""
        if self.strategy == "who_form":
            shown_answers = [
                QuestionAnswer(a["id"], a["question"],
                               shown(self.form.by_id[a["id"]], a["answer"]), a["evidence"])
                for a in answers["answers"] if a["source"] != "profile"]
        else:
            shown_answers = [QuestionAnswer(**a) for a in answers]
        return format_answers(shown_answers, transcript)


def create_questionnaire(llm_client, config) -> Questionnaire:
    """Create the questionnaire according to inference.questionnaire settings."""
    bank = load_question_bank(config.bank or None)
    if config.strategy == "who_form":
        return Questionnaire("who_form", WhoAnswerer(llm_client, PROMPTS[config.who_prompt]),
                             form=load_who_form(config.who_bank or None), bank=bank)
    retriever = QuestionRetriever(bank, model_name=config.embedding_model,
                                  top_k=config.top_k,
                                  min_similarity=config.min_similarity,
                                  window=config.window)
    return Questionnaire("retrieval", QuestionAnswerer(llm_client, PROMPTS[config.prompt]),
                         retriever=retriever)
