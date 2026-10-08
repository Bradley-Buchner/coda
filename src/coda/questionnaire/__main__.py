"""Run questionnaire retrieval on a transcript.

Usage:

    python -m coda.questionnaire --text transcript.txt [--top-k 10]
        [--min-similarity 0.5] [--window 2] [--bank bank.tsv] [--json out.json]
"""
import argparse
import json
import sys

from coda.questionnaire.bank import load_question_bank
from coda.questionnaire.retriever import QuestionRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve questions from a bank that best align with a transcript.")
    parser.add_argument("--text", required=True,
                        help="Transcript text file, or - for stdin")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-similarity", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=2,
                        help="Sentences per dialogue window")
    parser.add_argument("--bank", default=None,
                        help="Question bank TSV (default: packaged WHO VA bank)")
    parser.add_argument("--json", default=None, help="Also write results to this file")
    args = parser.parse_args()

    text = sys.stdin.read() if args.text == "-" else open(args.text).read()
    retriever = QuestionRetriever(load_question_bank(args.bank),
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

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([{"id": r.question.id, "question": r.question.text,
                        "score": round(r.score, 4), "matched_text": r.matched_text,
                        "sibling_of": r.sibling_of}
                       for r in retrieved], fh, indent=2)


if __name__ == "__main__":
    main()
