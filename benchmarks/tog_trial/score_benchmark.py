"""Score the results of run_benchmark.py, without any LLM calls.

Prints the tables and writes them to <out>/summary.json. In the comparisons, a
negative log-loss difference means the first arm is better. The calculator arm
ranks the candidates by the ProbBase grades and the saved answers alone, so it
has no probabilities to score. Pilot results are for debugging only.

Cases in chain_cases.json's label_to_check are left out unless
--include-label-check is given.

Usage
-----
    python benchmarks/tog_trial/score_benchmark.py
    python benchmarks/tog_trial/score_benchmark.py --out results/other
"""
import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
SETS = ("pilot", "original", "sparse")
ARMS = ("baseline", "second_look", "findings_only", "facts_only", "full",
        "placebo", "calculator")
COMPARISONS = (("full", "baseline"), ("full", "second_look"),
               ("full", "placebo"), ("findings_only", "full"),
               ("facts_only", "full"), ("second_look", "baseline"))
FLOOR = 0.01  # Probability given to a true cause an arm left out, for log loss
BOOTSTRAPS = 10000
SEED = 0
GRADE_VALUES = {"I": 1.0, "A+": 0.8, "A": 0.5, "A-": 0.2}
FACT = re.compile(r"\((.*), (is_a|causes), (.*)\)")


def scores(causes):
    """Return {cause name: probability}, most likely first."""
    ranked = sorted(causes.values(), key=lambda c: c["score"], reverse=True)
    return {c["name"]: c["score"] for c in ranked}


def support(finding):
    """Return how much a checked finding supports its cause: its grade's value,
    positive if the case gives the graded answer, negative if the opposite, and
    0 if the case doesn't say.
    """
    if finding["status"] == "not_mentioned":
        return 0
    value = GRADE_VALUES[finding["grade"]]
    return value if finding["status"] == finding["graded_answer"] else -value


def calculator(candidates):
    """Rank the candidates by mean support per finding; ties keep their order.
    The values returned only give that order, and aren't probabilities.
    """
    def mean_support(candidate):
        findings = candidate["findings"] or []
        return sum(map(support, findings)) / len(findings) if findings else 0

    ranked = sorted(candidates, key=mean_support, reverse=True)
    return {c["cause"]: len(ranked) - i for i, c in enumerate(ranked)}


def top(prediction):
    return next(iter(prediction), "-")


def loss(prediction, label):
    return -math.log(max(prediction.get(label, 0), FLOOR))


def mcnemar(rescues, harms):
    """Return the exact two-sided McNemar p-value."""
    flips = rescues + harms
    if not flips:
        return 1.0
    tail = sum(comb(flips, k) for k in range(min(rescues, harms) + 1))
    return min(1.0, 2 * tail / 2 ** flips)


def csmf_accuracy(predictions, labels):
    """Return CSMF accuracy, from each case's top cause. Some allowed causes
    have no cases, so the rarest cause's true fraction is 0.
    """
    true = Counter(labels[case] for case in predictions)
    predicted = Counter(top(p) for p in predictions.values())
    total = len(predictions)
    error = sum(abs(true[c] - predicted[c]) / total for c in {*true, *predicted})
    return 1 - error / 2


def mean_move(prediction, reference):
    """Return half the summed probability change between two predictions."""
    return sum(abs(prediction.get(c, 0) - reference.get(c, 0))
               for c in {*prediction, *reference}) / 2


def arm_metrics(arm, predictions, baseline, labels, lenient):
    """Return an arm's measures over its cases, or None if it has none."""
    cases = [case for case in predictions if case in labels]
    if not cases:
        return None
    predictions = {case: predictions[case] for case in cases}
    right = {case: top(p) == labels[case] for case, p in predictions.items()}
    base_right = {case: top(baseline[case]) == labels[case] for case in cases
                  if case in baseline}
    rescues = sum(right[c] and not base_right[c] for c in base_right)
    harms = sum(base_right[c] and not right[c] for c in base_right)
    metrics = {
        "n": len(cases),
        "top1": sum(right.values()) / len(cases),
        "top3": sum(predictions[c].get(labels[c], 0) > 0 for c in cases)
        / len(cases),
        "lenient_top1": sum(top(predictions[c]) in
                            [labels[c], *lenient.get(labels[c], [])]
                            for c in cases) / len(cases),
        "csmf": csmf_accuracy(predictions, labels),
        "rescues": rescues,
        "harms": harms,
        "mcnemar_p": mcnemar(rescues, harms),
        "changed_top": sum(top(predictions[c]) != top(baseline[c])
                           for c in base_right),
    }
    if arm == "calculator":
        return metrics
    confidence = {True: [], False: []}
    for case, prediction in predictions.items():
        confidence[right[case]].append(max(prediction.values(), default=0))
    return {
        **metrics,
        "log_loss": statistics.mean(loss(predictions[c], labels[c])
                                    for c in cases),
        "mean_move": statistics.mean(mean_move(predictions[c], baseline[c])
                                     for c in base_right)
        if base_right else None,
        "confidence_right": statistics.mean(confidence[True])
        if confidence[True] else None,
        "confidence_wrong": statistics.mean(confidence[False])
        if confidence[False] else None,
    }


def paired_difference(first, second, labels):
    """Return the mean log-loss difference, first minus second, on their shared
    cases, with a bootstrap 95% interval.
    """
    cases = [c for c in first if c in second and c in labels]
    if not cases:
        return None
    differences = [loss(first[c], labels[c]) - loss(second[c], labels[c])
                   for c in cases]
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(differences, k=len(differences)))
                   for _ in range(BOOTSTRAPS))
    return {"n": len(cases), "difference": statistics.fmean(differences),
            "low": means[int(0.025 * BOOTSTRAPS)],
            "high": means[int(0.975 * BOOTSTRAPS) - 1]}


def noise_measures(first_runs, final_runs, baseline, full):
    """Compare each cause's change from first opinion to final answer with how
    much the same calls vary when repeated, over the causes in either answer
    that changed at two decimals. Causes absent from a run count as 0.
    """
    changes, noises = [], []
    for case in full:
        firsts = [baseline[case]] + [run[case] for run in first_runs
                                     if case in run]
        finals = [full[case]] + [run[case] for run in final_runs if case in run]
        if len(firsts) < 2 or len(finals) < 2:
            continue
        for cause in {*baseline[case], *full[case]}:
            change = abs(full[case].get(cause, 0) - baseline[case].get(cause, 0))
            if round(change, 2) == 0:
                continue
            values = [[run.get(cause, 0) for run in runs]
                      for runs in (firsts, finals)]
            changes.append(change)
            noises.append(max(max(v) - min(v) for v in values))
    if not changes:
        return None
    return {"changed_causes": len(changes),
            "median_change": statistics.median(changes),
            "median_noise": statistics.median(noises),
            "beyond_noise": sum(c > n for c, n in zip(changes, noises))
            / len(changes)}


def evidence_measures(full):
    """Return how the full arm's evidence was found, used and cited."""
    answers, kept_facts = Counter(), []
    fact_relations, directions = Counter(), Counter()
    made, kept, flags, rule_changed_top, unquoted = 0, 0, 0, 0, 0
    separating, findings_cited, unique_facts, facts_cited = 0, 0, 0, 0
    for saved in full.values():
        evidence = saved["result"]["kg_evidence"]
        candidates = evidence["candidates"]
        check = next((e for e in saved["trace"]
                      if e["step"] == "finding_check"), None)
        if check:
            unquoted += sum(item["status"] != "not_mentioned"
                            for item in check["response"]["findings"]) - len(
                {f["finding"] for c in candidates for f in c["findings"] or []
                 if f["status"] != "not_mentioned"})
        final = next(e for e in saved["trace"]
                     if e["step"] == "final_ranking")["response"]
        asked = max(final["top_causes"], key=lambda c: c["probability"])
        rule_changed_top += asked["cause_name"] \
            != top(scores(saved["result"]["causes"]))
        made += len(final["cited_facts"]) + len(final["cited_findings"])
        kept += len(evidence["cited"])
        flags += saved["result"]["reasoning"].count("(Stated direction:")
        directions.update(u["direction"] for u in final["candidate_updates"])
        for candidate in candidates:
            kept_facts.append(len(candidate["triples"]))
            answers.update(f["status"] for f in candidate["findings"] or [])
        for item in evidence["cited"]:
            match = FACT.fullmatch(item)
            if match:
                facts_cited += 1
                fact_relations[match.group(2)] += 1
                unique_facts += sum(item in c["triples"]
                                    for c in candidates) == 1
                continue
            values = [next((support(f) for f in c["findings"] or []
                            if f["finding"] == item), 0) for c in candidates]
            findings_cited += 1
            separating += max(values) > min(values)
    cases = len(full)
    return {
        "cases": cases,
        "mean_facts_kept_per_candidate": statistics.mean(kept_facts)
        if kept_facts else 0,
        "answers": dict(answers),
        "answers_rejected_for_their_quote": unquoted,
        "citations_made": made,
        "citations_kept": kept,
        "cited_facts": dict(fact_relations),
        "cited_facts_on_one_candidate": unique_facts,
        "cited_findings": findings_cited,
        "cited_findings_separating_candidates": separating,
        "cases_citing_a_fact": sum(
            any(FACT.fullmatch(i) for i in s["result"]["kg_evidence"]["cited"])
            for s in full.values()),
        "directions": dict(directions),
        "stated_direction_flags": flags,
        # Top cause differs from the LLM's, usually from the no_change rule
        "cases_with_top_cause_changed_by_code": rule_changed_top,
    }


def coverage_by_cause(full):
    """Return, per true cause, its candidate's kept facts and finding answers."""
    rows = defaultdict(lambda: {"cases": 0, "facts": 0, "yes": 0, "no": 0,
                                "not_mentioned": 0})
    for saved in full.values():
        for candidate in saved["result"]["kg_evidence"]["candidates"]:
            if candidate["cause"] != saved["label"]:
                continue
            row = rows[saved["label"]]
            row["cases"] += 1
            row["facts"] += len(candidate["triples"])
            for finding in candidate["findings"] or []:
                row[finding["status"]] += 1
    return dict(sorted(rows.items()))


def cost_measures(full):
    """Return the full arm's mean cost per case: the calls, characters and tokens
    it adds to the first opinion's one call, and the seconds of the whole run.
    Tokens are estimated at four characters each.
    """
    if not full:
        return None
    calls, characters, seconds = [], [], []
    for saved in full.values():
        steps = [e for e in saved["trace"] if "user_prompt" in e]
        calls.append(len(steps))
        characters.append(sum(len(e["system_prompt"]) + len(e["user_prompt"])
                              + len(json.dumps(e["response"])) for e in steps))
        seconds.append(saved["result"]["timings"]["inference_s"])
    return {"added_calls": statistics.mean(calls),
            "added_characters": statistics.mean(characters),
            "added_estimated_tokens": statistics.mean(characters) / 4,
            "total_seconds": statistics.mean(seconds)}


def show_table(rows):
    """Print dicts as columns, in the order of the first row's keys."""
    if not rows:
        print("  (none)")
        return

    def cell(value):
        if value is None:
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    keys = list(rows[0])
    cells = [keys] + [[cell(row.get(key)) for key in keys] for row in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(keys))]
    for line in cells:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(line, widths)))


def load_arm(path):
    return {file.stem: json.loads(file.read_text())
            for file in sorted(path.glob("*.json"))}


def score_set(out, set_name, labels, groups, lenient):
    """Return all the measures of one set, or None if it has no results."""
    folder = out / set_name
    full = {c: s for c, s in load_arm(folder / "full").items() if c in labels}
    if not full:
        return None
    baseline = {c: scores(s["trace"][0]["response"]["causes"])
                for c, s in full.items()}
    predictions = {"baseline": baseline, "calculator": {
        c: calculator(s["result"]["kg_evidence"]["candidates"])
        for c, s in full.items()}}
    for arm in ("second_look", "findings_only", "facts_only", "full",
                "placebo"):
        predictions[arm] = {
            c: scores(s["result"]["causes"])
            for c, s in load_arm(folder / arm).items() if c in labels}
    first_runs = [{c: scores(s["result"]["causes"])
                   for c, s in load_arm(folder / f"first_rep{i}").items()}
                  for i in (1, 2)]
    final_runs = [{c: scores(s["result"]["causes"])
                   for c, s in load_arm(folder / f"full_rep{i}").items()}
                  for i in (1, 2)]

    def table(subset):
        chosen = {c: labels[c] for c in subset}
        return {arm: arm_metrics(arm, {c: p for c, p in predictions[arm].items()
                                       if c in chosen}, baseline, chosen,
                                 lenient) for arm in ARMS}

    flips = [{"case": c, "label": labels[c], "baseline": top(baseline[c]),
              "full": top(predictions["full"][c]),
              "result": "rescue" if top(predictions["full"][c]) == labels[c]
              else "harm" if top(baseline[c]) == labels[c] else "both wrong"}
             for c in predictions["full"]
             if top(predictions["full"][c]) != top(baseline[c])]
    return {
        "all": table(list(full)),
        "groups": {name: table(cases) for name, cases in groups.items()
                   if any(c in full for c in cases)},
        "comparisons": {f"{a} vs {b}": paired_difference(
            predictions[a], predictions[b], labels) for a, b in COMPARISONS},
        "flips": flips,
        "noise": noise_measures(first_runs, final_runs, baseline,
                                predictions["full"]),
        "evidence": evidence_measures(full),
        "coverage": coverage_by_cause(full),
        "cost": cost_measures(full),
    }


def show_arms(arms, columns):
    show_table([{"arm": arm, **{k: m.get(k) for k in columns}}
                for arm, m in arms.items() if m])


def show_set(set_name, results):
    main_columns = ("n", "top1", "top3", "lenient_top1", "log_loss", "csmf",
                    "rescues", "harms", "mcnemar_p", "changed_top",
                    "mean_move", "confidence_right", "confidence_wrong")
    print(f"\n{'=' * 60}\n{set_name.upper()}"
          + (" (debugging only)" if set_name == "pilot" else "")
          + f"\n{'=' * 60}")
    print("\nArms")
    show_arms(results["all"], main_columns)
    counts = {arm: m["n"] if m else 0 for arm, m in results["all"].items()}
    if len(set(counts.values())) > 1:
        print(f"  WARNING: arms cover different cases, so compare them only in "
              f"the differences below: {counts}")
    print("\nLog-loss differences (negative: the first arm is better)")
    show_table([{"comparison": name, **values}
                for name, values in results["comparisons"].items() if values])
    print("\nTop-cause flips, Full ToG vs baseline")
    show_table(results["flips"])
    print("\nBy group, baseline and Full ToG")
    show_table([{"group": name, "arm": arm, **{k: arms[arm].get(k) for k in
                                               ("n", "top1", "log_loss")}}
                for name, arms in results["groups"].items()
                for arm in ("baseline", "full") if arms[arm]])
    print("\nRun-to-run noise")
    show_table([results["noise"]] if results["noise"] else [])
    print("\nEvidence")
    print(json.dumps(results["evidence"], indent=2))
    print("\nEvidence for the true cause, when it's a candidate")
    show_table([{"cause": cause, **row}
                for cause, row in results["coverage"].items()])
    print("\nFull ToG cost per case")
    show_table([results["cost"]] if results["cost"] else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(HERE / "results" / "benchmark"))
    parser.add_argument("--include-label-check", action="store_true",
                        help="Keep the cases whose labels await a check")
    args = parser.parse_args()

    out = Path(args.out)
    plan = json.loads((HERE / "chain_cases.json").read_text())
    excluded = [] if args.include_label_check else list(plan["label_to_check"])
    cases = json.loads((HERE / "cases.json").read_text())
    pilot = plan["pilot"]
    # The same ids and labels apply to the sparse versions
    test = [c for c in cases if c["case_id"] not in pilot
            and c["case_id"] not in excluded]
    sets = {"pilot": [c for c in cases if c["case_id"] in pilot],
            "original": test, "sparse": test}

    info_path = out / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    print(f"Run: {info}")
    if info.get("dirty"):
        print("WARNING: these results came from uncommitted code.")
    print(f"Left out, awaiting a label check: {excluded or 'none'}")

    summary = {"run": info, "excluded": excluded}
    for set_name in SETS:
        labels = {c["case_id"]: c["cause_name"] for c in sets[set_name]}
        groups = {"chain": [c for c in plan["chain"] if c in labels]}
        for key in ("cause_name", "age_group"):
            for case in sets[set_name]:
                groups.setdefault(case[key], []).append(case["case_id"])
        results = score_set(out, set_name, labels, groups, plan["lenient"])
        if results:
            summary[set_name] = results
            show_set(set_name, results)

    failures = []
    failure_path = out / "failed_calls.jsonl"
    if failure_path.exists():
        failures = [json.loads(line)
                    for line in failure_path.read_text().splitlines()]
    summary["failed_attempts"] = Counter(
        f"{f['set']}/{f['arm']}" for f in failures)
    print(f"\nFailed attempts, not saved: {dict(summary['failed_attempts'])}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
