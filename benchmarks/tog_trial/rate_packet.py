"""Rate the review packet in the terminal, one item at a time.

Each answer is saved to the packet's CSV file right away, so you can stop at
any time and pick up where you left off. At any prompt:

    Enter   keep the current answer, or leave it blank
    <       go back to the previous item
    q       save and quit

Usage
-----
    python benchmarks/tog_trial/rate_packet.py compare
    python benchmarks/tog_trial/rate_packet.py evidence
    python benchmarks/tog_trial/rate_packet.py findings
    python benchmarks/tog_trial/rate_packet.py compare --start 12
"""
import argparse
import re
import shutil
import textwrap
from itertools import groupby
from pathlib import Path

from make_review_packet import (CORRECT_ANSWER, EVIDENCE, HERE, MORE_USEFUL,
                                YES_NO, read_csv, write_csv)

CHOICES = {
    "more_useful": ({**MORE_USEFUL, "s": "same"}, "A, B or same (s)"),
    "yes_no": (YES_NO, "yes or no (y/n)"),
    "rating": ({**EVIDENCE, "u": "useful", "f": "filler", "m": "misleading"},
               "useful, filler or misleading (u/f/m)"),
    "correct_answer": ({**CORRECT_ANSWER, "nm": "not_mentioned",
                        "u": "unclear"},
                       "yes, no, not_mentioned or unclear (y/n/nm/u)"),
}


class Back(Exception):
    pass


class Quit(Exception):
    pass


def show(text):
    width = min(shutil.get_terminal_size().columns, 100)
    for line in text.split("\n"):
        print(textwrap.fill(line, width, subsequent_indent="  "
                            if line.startswith("- ") else "") or "")


def ask(label, kind, current):
    """Return the answer to one prompt, which may be blank or free text."""
    accepted, hint = CHOICES.get(kind, ({}, "free text"))
    while True:
        shown = f" [{current}]" if current else ""
        value = input(f"{label}: {hint}{shown} > ").strip()
        if value == "<":
            raise Back
        if value.lower() == "q":
            raise Quit
        if not value or not accepted:
            return value or current
        key = value.lower().replace(" ", "_").replace("-", "_")
        if key in accepted:
            return accepted[key]
        print(f"  Not one of: {hint}")


def comparison_units(review):
    text = (review / "1_comparison.md").read_text()
    items = dict(re.findall(r"## Item (\d+)\n(.*?)(?=\n## Item |\Z)", text,
                            re.S))
    rows = read_csv(review / "1_ratings.csv")
    units = []
    for row in rows:
        text = items[row["item"]]
        names = dict(re.findall(r"\*\*Output ([AB]): (.+?)\*\*", text))
        prompts = [("More useful", "more_useful", "more_useful")]
        prompts += [(f"Wrong or unsupported claim in {side} ({names[side]})",
                     "yes_no", f"wrong_or_unsupported_in_{side}")
                    for side in "AB"]
        prompts.append(("Notes", "notes", "notes"))
        units.append((f"Item {row['item']}", text.replace("**", ""),
                      [(label, kind, row, column)
                       for label, kind, column in prompts]))
    return review / "1_ratings.csv", rows, units


def evidence_units(review):
    rows = read_csv(review / "2_evidence_check.csv")
    units = []
    for (set_name, case), group in groupby(rows,
                                           lambda r: (r["set"], r["case"])):
        group = list(group)
        reasoning = next(r for r in group if r["kind"] == "reasoning")
        cited = [r for r in group if r["kind"] == "cited item"]
        text = "\n".join([reasoning["narrative"], "", "ToG's reasoning:",
                          reasoning["text"], "", "Cited items:"]
                         + [f"- {n}. {r['text']}" for n, r in
                            enumerate(cited, 1)])
        prompts = [(f"Cited item {n}", "rating", r, "rating")
                   for n, r in enumerate(cited, 1)]
        prompts.append(("Does the reasoning add anything the narrative "
                        "alone doesn't?", "yes_no", reasoning,
                        "adds_beyond_narrative"))
        units.append((f"{case} ({set_name})", text, prompts))
    return review / "2_evidence_check.csv", rows, units


def finding_units(review):
    rows = read_csv(review / "3_finding_checks.csv")
    units = [(f"Item {row['item']}",
              f"{row['narrative']}\n\nQuestion: {row['question']}",
              [("Correct answer, from the narrative alone", "correct_answer",
                row, "correct_answer")])
             for row in rows]
    return review / "3_finding_checks.csv", rows, units


PARTS = {"compare": comparison_units, "evidence": evidence_units,
         "findings": finding_units}


def done(unit):
    return all(row[column] for _, kind, row, column in unit[2]
               if kind != "notes")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("part", choices=PARTS)
    parser.add_argument("--out", default=str(HERE / "results" / "benchmark"))
    parser.add_argument("--start", type=int,
                        help="Start at this item number, counting from 1")
    args = parser.parse_args()

    path, rows, units = PARTS[args.part](Path(args.out) / "review")
    columns = list(rows[0])
    index = args.start - 1 if args.start else next(
        (i for i, unit in enumerate(units) if not done(unit)), len(units))
    try:
        while index < len(units):
            title, text, prompts = units[index]
            finished = sum(map(done, units))
            print("\n" + "=" * 60)
            print(f"{title}  ({index + 1} of {len(units)}, "
                  f"{finished} complete)")
            print("=" * 60 + "\n")
            show(text)
            print()
            try:
                for label, kind, row, column in prompts:
                    row[column] = ask(label, kind, row[column] or "")
                    write_csv(path, columns, rows)
                index += 1
            except Back:
                index = max(index - 1, 0)
    except (Quit, EOFError, KeyboardInterrupt):
        pass
    write_csv(path, columns, rows)
    left = sum(not done(unit) for unit in units)
    print(f"\nSaved to {path}. {left} of {len(units)} items still need an "
          "answer." if left else f"\nSaved to {path}. All items answered.")


if __name__ == "__main__":
    main()
