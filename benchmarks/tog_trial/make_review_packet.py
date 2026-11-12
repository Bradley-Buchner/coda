"""Write the qualitative review packet from run_benchmark.py's results.

Writes these files to <out>/review, without any LLM calls:

    1_comparison.md and 1_ratings.csv
        30 cases, plus 5 hidden repeats, with the prompted agent and ToG shown
        as A and B in random order, each labeled. ToG's reasoning leaves out
        the sentences code adds about probability changes.
    2_evidence_check.csv
        The same 30 cases: rate each of ToG's cited items and its reasoning.
    3_finding_checks.csv
        60 sampled findings, to answer from the narrative alone.
    key.json
        Which output is which, and the LLM's answers to the finding checks.
        Don't open it before rating.

--score reads the filled CSV files and prints the results. Blank cells count as
unrated, and any value outside those listed in the README stops the scoring.

Usage
-----
    python benchmarks/tog_trial/make_review_packet.py
    python benchmarks/tog_trial/make_review_packet.py --score
"""
import argparse
import csv
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261107
NUM_CASES = 30
NUM_REPEATS = 5
FINDINGS_PER_ANSWER = 20
ANSWERS = ("yes", "no", "not_mentioned")
RATING_COLUMNS = ["item", "more_useful", "wrong_or_unsupported_in_A",
                  "wrong_or_unsupported_in_B", "notes"]
# Accepted ratings, after lowering case and joining words with underscores
MORE_USEFUL = {"a": "A", "b": "B", "same": "same"}
YES_NO = {"yes": "yes", "y": "yes", "no": "no", "n": "no"}
EVIDENCE = {"useful": "useful", "filler": "filler", "misleading": "misleading"}
CORRECT_ANSWER = {**YES_NO, "not_mentioned": "not_mentioned",
                  "unclear": "unclear"}
AGENT_NAMES = {"baseline": "Prompted agent", "tog": "ToG"}


def top(causes):
    return max(causes.values(), key=lambda c: c["score"])["name"]


def load_full(out, set_name, skip):
    folder = out / set_name / "full"
    return {path.stem: json.loads(path.read_text())
            for path in sorted(folder.glob("*.json")) if path.stem not in skip}


def select_cases(results, rng):
    """Return NUM_CASES (set, case id) pairs: every flip of the top cause first,
    then originals taken in turn across the causes.
    """
    flips = [(set_name, case) for set_name, saved in results.items()
             for case, s in saved.items()
             if top(s["trace"][0]["response"]["causes"])
             != top(s["result"]["causes"])]
    rng.shuffle(flips)
    chosen = flips[:NUM_CASES]
    by_cause = defaultdict(list)
    for case, saved in sorted(results["original"].items()):
        if ("original", case) not in chosen:
            by_cause[saved["label"]].append(("original", case))
    for cases in by_cause.values():
        rng.shuffle(cases)
    causes = sorted(by_cause)
    rng.shuffle(causes)
    while len(chosen) < NUM_CASES and any(by_cause.values()):
        for cause in causes:
            if by_cause[cause] and len(chosen) < NUM_CASES:
                chosen.append(by_cause[cause].pop())
    return chosen


def tog_reasoning(saved):
    """Return ToG's reasoning without the sentences code adds about changes."""
    final = next((e for e in saved["trace"] if e["step"] == "final_ranking"),
                 None)
    return final["response"]["reasoning"] if final else \
        saved["result"]["reasoning"]


def show_output(name, agent, causes, reasoning):
    ranked = sorted(causes.values(), key=lambda c: c["score"], reverse=True)
    lines = [f"**Output {name}: {AGENT_NAMES[agent]}**", ""]
    lines += [f"- {c['name']}: {c['score']:.2f}" for c in ranked]
    return lines + ["", reasoning, ""]


def write_csv(path, columns, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Return a sheet's rows that have any value, whether a spreadsheet program
    saved it as UTF-8, with or without a byte-order mark, or as Windows text.
    """
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    rows = csv.DictReader(io.StringIO(text, newline=""))
    return [row for row in rows if any(row.values())]


def rating(row, column, accepted):
    """Return a cell's rating, or '' if blank. Exit if it isn't accepted."""
    value = (row[column] or "").strip().lower().replace(" ", "_") \
        .replace("-", "_")
    if value and value not in accepted:
        sys.exit(f"Item {row.get('item') or row.get('case')}: {column} is "
                 f"{row[column]!r}, expected one of {sorted(accepted)}")
    return accepted.get(value, "")


def write_packet(results, review):
    rng = random.Random(SEED)
    chosen = select_cases(results, rng)
    repeats = rng.sample(chosen, min(NUM_REPEATS, len(chosen)))
    items = chosen + repeats
    rng.shuffle(items)

    lines = ["# Comparison", "",
             "For each item, read the narrative and both outputs. Then fill in "
             "`1_ratings.csv`: `more_useful` is A, B or same; the next two "
             "columns are yes or no. Rate each item on its own.", ""]
    key = {"items": {}}
    ratings = []
    for number, (set_name, case) in enumerate(items, 1):
        saved = results[set_name][case]
        baseline = saved["trace"][0]["response"]
        outputs = [("baseline", baseline["causes"], baseline["reasoning"]),
                   ("tog", saved["result"]["causes"], tog_reasoning(saved))]
        rng.shuffle(outputs)
        lines += [f"## Item {number}", "", saved["narrative"], ""]
        for name, output in zip("AB", outputs):
            lines += show_output(name, *output)
        key["items"][str(number)] = {
            "set": set_name, "case": case, "A": outputs[0][0],
            "B": outputs[1][0],
            "flip": top(baseline["causes"]) != top(saved["result"]["causes"])}
        ratings.append({"item": number})
    (review / "1_comparison.md").write_text("\n".join(lines))
    write_csv(review / "1_ratings.csv", RATING_COLUMNS, ratings)

    rows = []
    for set_name, case in chosen:
        saved = results[set_name][case]
        rows.append({"set": set_name, "case": case, "kind": "reasoning",
                     "text": saved["result"]["reasoning"],
                     "narrative": saved["narrative"]})
        rows += [{"set": set_name, "case": case, "kind": "cited item",
                  "text": item}
                 for item in saved["result"]["kg_evidence"]["cited"]]
    write_csv(review / "2_evidence_check.csv",
              ["set", "case", "kind", "text", "rating",
               "adds_beyond_narrative", "narrative"], rows)

    pools = defaultdict(list)
    for case, saved in results["original"].items():
        for candidate in saved["result"]["kg_evidence"]["candidates"]:
            for finding in candidate["findings"] or []:
                pools[finding["status"]].append(
                    (case, finding["finding"], finding["status"]))
    sampled = [row for answer in ANSWERS for row in rng.sample(
        sorted(pools[answer]), min(FINDINGS_PER_ANSWER, len(pools[answer])))]
    rng.shuffle(sampled)
    write_csv(review / "3_finding_checks.csv",
              ["item", "case", "narrative", "question", "correct_answer"],
              [{"item": i, "case": case, "question": question,
                "narrative": results["original"][case]["narrative"]}
               for i, (case, question, _) in enumerate(sampled, 1)])
    key["findings"] = {str(i): status
                       for i, (_, _, status) in enumerate(sampled, 1)}
    key["finding_pools"] = {answer: len(pools[answer]) for answer in ANSWERS}
    (review / "key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote the packet to {review}: {len(items)} comparison items, "
          f"{len(sampled)} finding checks.")


def score_packet(review):
    key = json.loads((review / "key.json").read_text())
    rows = read_csv(review / "1_ratings.csv")
    votes = defaultdict(list)
    rated = {}
    for row in rows:
        item = key["items"][row["item"]]
        choice = rating(row, "more_useful", MORE_USEFUL)
        wrong = {item[side]: rating(row, f"wrong_or_unsupported_in_{side}",
                                    YES_NO) for side in "AB"}
        if choice:
            prefers = item.get(choice, "same")
            votes[item["set"], item["case"]].append(prefers)
            # A hidden repeat only checks consistency
            rated.setdefault((item["set"], item["case"]),
                             {"flip": item["flip"], "prefers": prefers,
                              "wrong": wrong})
    print(f"Comparison items unrated: "
          f"{sum(not (r['more_useful'] or '').strip() for r in rows)} "
          f"of {len(rows)}")
    groups = {"All cases": list(rated.values()),
              "Top cause flipped": [r for r in rated.values() if r["flip"]],
              "Top cause unchanged": [r for r in rated.values()
                                      if not r["flip"]]}
    for name, group in groups.items():
        if not group:
            continue
        preference = Counter(r["prefers"] for r in group)
        wrong = {agent: "{} of {}".format(
            sum(r["wrong"][agent] == "yes" for r in group),
            sum(bool(r["wrong"][agent]) for r in group))
            for agent in ("tog", "baseline")}
        print(f"{name}: of {len(group)} cases, more useful {dict(preference)}, "
              f"so prefer ToG in {preference['tog'] / len(group):.0%}; "
              f"wrong or unsupported claim {wrong}")
    pairs = [v for v in votes.values() if len(v) > 1]
    print(f"Hidden repeats rated the same way: "
          f"{sum(len(set(v)) == 1 for v in pairs)} of {len(pairs)}")

    evidence = read_csv(review / "2_evidence_check.csv")
    print("Cited items:", dict(Counter(
        rating(r, "rating", EVIDENCE) or "unrated" for r in evidence
        if r["kind"] == "cited item")))
    print("Reasoning adds beyond the narrative:", dict(Counter(
        rating(r, "adds_beyond_narrative", YES_NO) or "unrated"
        for r in evidence if r["kind"] == "reasoning")))

    totals, right, unclear = Counter(), Counter(), 0
    for row in read_csv(review / "3_finding_checks.csv"):
        status = key["findings"][row["item"]]
        answer = rating(row, "correct_answer", CORRECT_ANSWER)
        unclear += answer == "unclear"
        if answer and answer != "unclear":
            totals[status] += 1
            right[status] += answer == status
    print("Finding checks correct, by the LLM's answer:",
          {a: f"{right[a]} of {totals[a]}" for a in ANSWERS},
          f"({unclear} unclear, left out)")
    # Answers are sampled equally, so weight them by how often the LLM gave each
    pools = {a: key["finding_pools"][a] for a in ANSWERS if totals[a]}
    if pools:
        overall = sum(pools[a] * right[a] / totals[a] for a in pools) \
            / sum(pools.values())
        print(f"Overall, weighted by how often the LLM gives each answer: "
              f"{overall:.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(HERE / "results" / "benchmark"))
    parser.add_argument("--include-label-check", action="store_true",
                        help="Allow the cases whose labels await a check")
    parser.add_argument("--score", action="store_true",
                        help="Score the filled CSV files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing packet and its ratings")
    args = parser.parse_args()

    out = Path(args.out)
    review = out / "review"
    if args.score:
        score_packet(review)
        return
    if (review / "key.json").exists() and not args.overwrite:
        sys.exit(f"{review} already holds a packet, which may have your "
                 "ratings. Use --overwrite to replace it.")
    plan = json.loads((HERE / "chain_cases.json").read_text())
    skip = set() if args.include_label_check else set(plan["label_to_check"])
    results = {set_name: load_full(out, set_name, skip)
               for set_name in ("original", "sparse")}
    review.mkdir(parents=True, exist_ok=True)
    write_packet(results, review)


if __name__ == "__main__":
    main()
