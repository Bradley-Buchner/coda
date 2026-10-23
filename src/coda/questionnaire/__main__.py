"""Answer the questionnaire on a transcript.

Usage:

    python -m coda.questionnaire --text transcript.txt [--strategy who_form|retrieval]
        [--age-days DAYS] [--sex SEX] [--stillbirth] [--top-k 10] [--min-similarity 0.5]
        [--window 2] [--bank bank.tsv] [--who-bank bank.json] [--json out.json] [--answer]
        [--provider PROVIDER] [--model MODEL]

The who_form strategy asks the WHO 2016 VA form's questions in rounds, following the
form's skip logic.

The retrieval strategy lists the bank questions most similar to the
transcript, and answers them with --answer.

Defaults come from inference.questionnaire and inference.llm in settings.yaml.
"""
import argparse
import json
import sys

from coda.config import PROMPTS, settings
from coda.questionnaire.answerer import QuestionAnswerer
from coda.questionnaire.bank import load_question_bank
from coda.questionnaire.retriever import QuestionRetriever
from coda.questionnaire.who_answerer import WhoAnswerer, who_record
from coda.questionnaire.who_form import load_who_form


def _llm_client(args):
    from coda.llm_api import create_llm_client
    return create_llm_client(provider=args.provider or settings.inference.llm.provider,
                             model=args.model or settings.inference.llm.model)


def _print_answers(answers):
    for a in answers:
        answer = a.answer if isinstance(a.answer, str) else json.dumps(a.answer)
        print(f"{answer:>8}  {a.id}  {a.question}")
        if a.evidence:
            print(f"           evidence: {a.evidence}")


def main():
    config = settings.inference.questionnaire
    parser = argparse.ArgumentParser(
        description="Answer standardized VA questions from a transcript.")
    parser.add_argument("--text", required=True,
                        help="Transcript text file, or - for stdin")
    parser.add_argument("--strategy", choices=("who_form", "retrieval"),
                        default=config.strategy,
                        help="who_form: the WHO 2016 form's questions and skip logic; "
                             "retrieval: the questions most similar to the transcript")
    parser.add_argument("--age-days", type=float, default=None,
                        help="Age at death in days, for who_form (default: asked)")
    parser.add_argument("--sex", default=None, choices=("male", "female", "undetermined"),
                        help="Sex of the deceased, for who_form (default: asked)")
    parser.add_argument("--stillbirth", action="store_true",
                        help="The deceased was stillborn, for who_form")
    parser.add_argument("--top-k", type=int, default=config.top_k)
    parser.add_argument("--min-similarity", type=float, default=config.min_similarity)
    parser.add_argument("--window", type=int, default=config.window,
                        help="Sentences per dialogue window")
    parser.add_argument("--bank", default=config.bank or None,
                        help="Question bank TSV (default: packaged WHO VA bank)")
    parser.add_argument("--who-bank", default=config.who_bank or None,
                        help="WHO-form bank JSON (default: packaged WHO 2016 form bank)")
    parser.add_argument("--json", default=None, help="Also write results to this file")
    parser.add_argument("--answer", action="store_true",
                        help="Also answer the retrieved questions with the LLM")
    parser.add_argument("--provider", default=None,
                        help="LLM provider (default: inference.llm.provider)")
    parser.add_argument("--model", default=None,
                        help="LLM model (default: inference.llm.model)")
    args = parser.parse_args()

    text = sys.stdin.read() if args.text == "-" else open(args.text).read()
    questions = load_question_bank(args.bank)
    output = {}

    if args.strategy == "who_form":
        answerer = WhoAnswerer(_llm_client(args), PROMPTS[config.who_prompt])
        answers, interview, complete = answerer.answer_form(
            text, load_who_form(args.who_bank), args.age_days, args.sex, args.stillbirth)
        record = who_record(answers, interview, questions)
        _print_answers(answers)
        print(f"\n{len(answers)} WHO questions in {len(interview.rounds)} rounds; "
              f"{len(record['items'])} probbase items"
              + ("" if complete else "; STOPPED EARLY: an LLM call failed"))
        output.update(record, complete=complete)
    else:
        retriever = QuestionRetriever(questions,
                                      model_name=config.embedding_model,
                                      top_k=args.top_k,
                                      min_similarity=args.min_similarity,
                                      window=args.window)
        retrieved = retriever.retrieve(text)
        for rank, r in enumerate(retrieved, 1):
            sibling = f"  (option of {r.sibling_of})" if r.sibling_of else ""
            print(f"{rank:2d}. {r.score:.3f}  {r.question.id}  {r.question.text}{sibling}")
            print(f"           matched: {r.matched_text}")
        if not retrieved:
            print("No questions above the similarity threshold.")
        output["retrieved"] = [{"id": r.question.id, "question": r.question.text,
                                "score": round(r.score, 4),
                                "matched_text": r.matched_text,
                                "sibling_of": r.sibling_of} for r in retrieved]
        answers = []
        if args.answer and retrieved:
            answerer = QuestionAnswerer(_llm_client(args), PROMPTS[config.prompt])
            answers = answerer.answer(text, [r.question for r in retrieved])
            print()
            _print_answers(answers)
        output["answers"] = [a.to_dict() for a in answers]

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(output, fh, indent=2)


if __name__ == "__main__":
    main()
