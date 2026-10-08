"""Retrieve the bank questions that best align with a piece of dialogue.

Dialogue text is split into short overlapping windows of consecutive sentences,
and every window and every question is embedded with the shared encoder
(coda.embeddings). A question's score is its best cosine similarity to any
single window. A high scores suggests that a question is relevant to some part of
the dialogue.

Since embedding similarity struggles to tell an item's options apart
(e.g. fever durations), when one option is retrieved, its other options are added
too. For "he had a fever for five days", retrieval may find "Did the fever last at
least 2 weeks?" but miss "Did the fever last less than a week?"; with both present,
the answer can be "no" to the first and "yes" to the second.
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence

import numpy as np

from coda.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from coda.questionnaire.bank import Question

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_units(text: str, window: int = 2, max_words: int = 40) -> List[str]:
    """Split text into overlapping windows of `window` consecutive sentences.

    Sentences longer than `max_words` are cut into pieces first, so a window
    stays well within the embedding model's input limit (~190 words).
    """
    sentences = []
    for sentence in SENTENCE_BREAK.split(text or ""):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            sentences.append(" ".join(words[start:start + max_words]))
    if len(sentences) <= window:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i:i + window])
            for i in range(len(sentences) - window + 1)]


@dataclass(frozen=True)
class RetrievedQuestion:
    question: Question
    score: float
    matched_text: str
    sibling_of: Optional[str] = None  # set when added as another option of this id


class QuestionRetriever:
    """Rank bank questions by their best cosine similarity to the dialogue."""

    def __init__(self, questions: Sequence[Question],
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 top_k: int = 10, min_similarity: float = 0.5, window: int = 2,
                 encoder=None):
        self.questions = list(questions)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.window = window
        self._encode = encoder or partial(embed_texts, model_name=model_name)
        self._question_vectors = self._encode([q.text for q in self.questions])
        self._options = defaultdict(list)
        for index, question in enumerate(self.questions):
            if question.group:
                self._options[question.group].append(index)

    def retrieve(self, text: str, top_k: Optional[int] = None,
                 min_similarity: Optional[float] = None) -> List[RetrievedQuestion]:
        top_k = self.top_k if top_k is None else top_k
        min_similarity = self.min_similarity if min_similarity is None \
            else min_similarity
        units = split_into_units(text, self.window)
        if not units:
            return []
        similarity = self._encode(units) @ self._question_vectors.T
        scores = similarity.max(axis=0)
        best_unit = similarity.argmax(axis=0)

        def result(j, sibling_of=None):
            return RetrievedQuestion(question=self.questions[j],
                                     score=float(scores[j]),
                                     matched_text=units[best_unit[j]],
                                     sibling_of=sibling_of)

        top = [j for j in np.argsort(-scores, kind="stable")[:top_k]
               if scores[j] >= min_similarity]
        seen = set(top)
        retrieved = []
        for j in top:
            retrieved.append(result(j))
            for option in self._options.get(self.questions[j].group, []):
                if option not in seen:
                    seen.add(option)
                    retrieved.append(result(option, sibling_of=self.questions[j].id))
        return retrieved
