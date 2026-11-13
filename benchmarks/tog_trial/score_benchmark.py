"""Score the results of run_benchmark.py, without any LLM calls.

Prints the tables and writes them to <out>/summary.json. In the comparisons, a
negative log-loss difference means the first arm is better. The calculator arm
ranks the candidates by the ProbBase grades and the saved answers alone, so it
has no probabilities to score. The findings_look arm, the findings-only agent, is
scored when its results exist, and so are the reasoning arms, whose steps are
also compared with those of reason_none. Pilot results are for debugging only.

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
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from coda.inference.tog.reasoning import (
    diff_steps,
    item_verdict,
    stable_changes,
)

SETS = ("pilot", "original", "sparse")
REASON_ARMS = ("reason_none", "reason_kg", "reason_kg_walks", "reason_placebo")
ARMS = ("baseline", "second_look", "findings_only", "findings_look",
        "facts_only", "full", "placebo", *REASON_ARMS, "calculator")
COMPARISONS = (("full", "baseline"), ("full", "second_look"),
               ("full", "placebo"), ("findings_only", "full"),
               ("facts_only", "full"), ("second_look", "baseline"),
               ("findings_look", "baseline"), ("findings_look", "second_look"),
               ("findings_look", "findings_only"), ("findings_look", "placebo"),
               ("findings_look", "full"), ("reason_kg", "reason_none"),
               ("reason_kg", "second_look"), ("reason_kg", "full"),
               ("reason_kg", "reason_placebo"),
               ("reason_kg_walks", "reason_kg"))
CHANGES = ("added", "omitted", "changed")
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
    """Return how much a checked finding supports its cause: its grade's value
    if the case gives the graded answer, and otherwise 0, since InterVA ignores
    the other answers.
    """
    if finding["status"] != finding["graded_answer"]:
        return 0
    return GRADE_VALUES[finding["grade"]]


def calculator(candidates):
    """Rank the candidates by the total support of their findings, so that, as
    in InterVA, other answers don't count against them; ties keep their order.
    The values returned only give that order, and aren't probabilities.
    """
    ranked = sorted(candidates, reverse=True, key=lambda candidate: sum(
        map(support, candidate["findings"] or [])))
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
        made += len(final.get("cited_facts", [])) + len(final["cited_findings"])
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


def cost_measures(full, seconds_key="total_seconds"):
    """Return an arm's mean cost per case: the calls, characters and tokens it
    adds to the first opinion's one call, and its timed seconds: the whole run
    for the full arm, only the added calls for findings_look. Tokens are
    estimated at four characters each.
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
            seconds_key: statistics.mean(seconds)}


def steps_of(saved):
    return saved["result"]["kg_evidence"].get("steps", [])


def items_of(saved):
    return saved["result"]["kg_evidence"].get("items", [])


def grounding_measures(runs, labels, chain):
    """Return how an arm's reasoning steps use the KG items shown."""
    steps = [s for saved in runs.values() for s in steps_of(saved)]
    if not steps:
        return None
    grounded = [s for s in steps if s["valid"]]
    narrative = [s for s in steps if s["quote_ok"] is not None]
    shown_rules = [i["id"] for saved in runs.values() for i in items_of(saved)
                   if i["kind"] == "R"]
    cited_rules = sum(len({i for s in steps_of(saved) for i in s["valid"]
                           if i.startswith("R")}) for saved in runs.values())
    underlying = {case: {s["underlying"] for s in steps_of(runs[case])
                         if s["role"] in ("sequence", "coding_rule")}
                  - {""} for case in chain if case in runs}
    return {
        "cases": len(runs),
        "steps_per_case": len(steps) / len(runs),
        "grounded": len(grounded) / len(steps),
        "with_invalid_id": sum(bool(s["invalid"]) for s in steps) / len(steps),
        "quote_ok": sum(s["quote_ok"] for s in narrative) / len(narrative)
        if narrative else None,
        "consistent": sum(s["check"] == "consistent" for s in grounded)
        / len(grounded) if grounded else None,
        "inconsistent": sum(s["check"] == "inconsistent" for s in grounded)
        / len(grounded) if grounded else None,
        "rejected": sum(s["check"] == "rejected" for s in grounded)
        / len(grounded) if grounded else None,
        "rules_cited": cited_rules / len(shown_rules) if shown_rules else None,
        "chain_underlying_right": sum(labels[c] in u
                                      for c, u in underlying.items()),
        "chain_underlying_wrong": sum(bool(u) and labels[c] not in u
                                      for c, u in underlying.items()),
        "chain_cases": len(underlying),
    }


def change_counts(diffs):
    """Return the share of cases with each kind of change and the mean number
    of steps per case, over {case: diff_steps result}.
    """
    row = {"cases": len(diffs)}
    for kind in (*CHANGES, "grounded"):
        row[f"{kind}_cases"] = sum(bool(d[kind]) for d in diffs.values()) \
            / len(diffs) if diffs else None
        row[f"{kind}_steps"] = statistics.mean(len(d[kind])
                                               for d in diffs.values()) \
            if diffs else None
    return row


def change_summary(kg_runs, control_runs):
    """Return how often an arm's reasoning changes from the control's: in the
    main runs, between repeat runs of either arm (noise), and in every run
    (stable), with the provable changes by the kinds of items they rest on.
    """
    kg, control = kg_runs[0], control_runs[0]
    shared = [c for c in kg if c in control]
    rows = {"raw": change_counts({c: diff_steps(steps_of(kg[c]),
                                                steps_of(control[c]))
                                  for c in shared})}
    noise = defaultdict(list)
    for runs in (kg_runs, control_runs):
        for i, first in enumerate(runs):
            for second in runs[i + 1:]:
                for c in shared:
                    if c in first and c in second:
                        noise[c].append(diff_steps(steps_of(first[c]),
                                                   steps_of(second[c])))
    rows["noise"] = {"cases": len(noise), **{
        f"{kind}_{unit}": statistics.mean(
            statistics.mean((bool if unit == "cases" else len)(d[kind])
                            for d in diffs) for diffs in noise.values())
        if noise else None
        for kind in CHANGES for unit in ("cases", "steps")}}
    complete = [c for c in shared if len(kg_runs) == len(control_runs) > 1
                and all(c in run for run in kg_runs + control_runs)]
    stable = {c: stable_changes([steps_of(run[c]) for run in kg_runs],
                                [steps_of(run[c]) for run in control_runs],
                                items_of(kg[c])) for c in complete}
    kinds = Counter()
    for changes in stable.values():
        for key in changes["provable_added"] + changes["provable_changed"]:
            kinds["+".join(changes["kinds"][key]) or "none"] += 1
    for name, prefix in (("stable", ""), ("provable", "provable_")):
        rows[name] = {"cases": len(stable), **{
            f"{kind}_{unit}": (statistics.mean(
                (bool if unit == "cases" else len)(s[prefix + kind])
                for s in stable.values()) if stable else None)
            for kind in CHANGES for unit in ("cases", "steps")}}
    return {"rows": rows, "provable_by_kind": dict(kinds)}


def placebo_tracking(placebo, real):
    """Return, for the placebo arm's steps citing a reversed rule or a moved
    finding, how often they agree with the item shown and with the real item.
    Links both ways aren't reversed, so they're left out.
    """
    def finding(item):
        return item["text"].split(" Typical of")[0]

    counts = Counter()
    for case, saved in placebo.items():
        if case not in real:
            continue
        shown = {i["id"]: i for i in items_of(saved)}
        real_items = items_of(real[case])
        by_id = {i["id"]: i for i in real_items}
        by_finding = {finding(i): i for i in real_items if i["kind"] == "F"}
        for step in steps_of(saved):
            for item_id in step["valid"]:
                item = shown[item_id]
                if item.get("both_ways"):
                    continue
                twin = by_finding.get(finding(item)) if item["kind"] == "F" \
                    else by_id.get(item_id)
                if item["kind"] == "W" or twin is None:
                    continue
                counts[item["kind"], "cited"] += 1
                counts[item["kind"], "follows_shown"] += \
                    item_verdict(item, step) == "consistent"
                counts[item["kind"], "follows_real"] += \
                    item_verdict(twin, step) == "consistent"
    return [{"kind": kind, "cited": counts[kind, "cited"],
             "follows_shown": counts[kind, "follows_shown"],
             "follows_real": counts[kind, "follows_real"]}
            for kind in ("R", "F") if counts[kind, "cited"]]


def reasoning_measures(folder, labels, chain):
    """Return the reasoning arms' grounding, change summaries against the
    control and placebo tracking, or None if they have no results.
    """
    def runs(arm):
        return [{c: s for c, s in load_arm(folder / name).items()
                 if c in labels}
                for name in (arm, f"{arm}_rep1", f"{arm}_rep2")
                if (folder / name).exists()]

    arms = {arm: runs(arm) for arm in REASON_ARMS}
    if not arms["reason_none"] or not arms["reason_kg"]:
        return None
    return {
        "grounding": {arm: grounding_measures(r[0], labels, chain)
                      for arm, r in arms.items() if r},
        "changes": {f"{arm} vs reason_none": change_summary(
            arms[arm], arms["reason_none"])
            for arm in REASON_ARMS[1:] if arms[arm]},
        "placebo_tracking": placebo_tracking(arms["reason_placebo"][0],
                                             arms["reason_kg"][0])
        if arms["reason_placebo"] else None,
    }


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


def score_set(out, set_name, labels, groups, lenient, reasoning=True):
    """Return all the measures of one set, or None if it has no results. The
    reasoning measures are left out if reasoning is False.
    """
    folder = out / set_name
    full = {c: s for c, s in load_arm(folder / "full").items() if c in labels}
    if not full:
        return None
    baseline = {c: scores(s["trace"][0]["response"]["causes"])
                for c, s in full.items()}
    predictions = {"baseline": baseline, "calculator": {
        c: calculator(s["result"]["kg_evidence"]["candidates"])
        for c, s in full.items()}}
    for arm in ("second_look", "findings_only", "findings_look", "facts_only",
                "full", "placebo", *REASON_ARMS):
        predictions[arm] = {
            c: scores(s["result"]["causes"])
            for c, s in load_arm(folder / arm).items() if c in labels}
    first_runs = [{c: scores(s["result"]["causes"])
                   for c, s in load_arm(folder / f"first_rep{i}").items()}
                  for i in (1, 2)]
    final_runs = [{c: scores(s["result"]["causes"])
                   for c, s in load_arm(folder / f"full_rep{i}").items()}
                  for i in (1, 2)]
    findings_runs = [{c: scores(s["result"]["causes"]) for c, s in
                      load_arm(folder / f"findings_look_rep{i}").items()}
                     for i in (1, 2)]
    findings_look = {c: s for c, s in load_arm(folder / "findings_look").items()
                     if c in labels}

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
        "findings_look": {
            "noise": noise_measures(first_runs, findings_runs, baseline,
                                    predictions["findings_look"]),
            "evidence": evidence_measures(findings_look),
            "cost": cost_measures(findings_look, "added_seconds"),
        } if findings_look else None,
        "reasoning": reasoning_measures(folder, labels, groups["chain"])
        if reasoning else None,
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
    counts = {arm: m["n"] for arm, m in results["all"].items() if m}
    if len(set(counts.values())) > 1:
        print(f"  WARNING: arms cover different cases, so compare them only in "
              f"the differences below: {counts}")
    print("\nLog-loss differences (negative: the first arm is better)")
    show_table([{"comparison": name, **values}
                for name, values in results["comparisons"].items() if values])
    print("\nTop-cause flips, Full ToG vs baseline")
    show_table(results["flips"])
    print("\nBy group, baseline, Full ToG and findings_look")
    show_table([{"group": name, "arm": arm, **{k: arms[arm].get(k) for k in
                                               ("n", "top1", "log_loss")}}
                for name, arms in results["groups"].items()
                for arm in ("baseline", "full", "findings_look") if arms[arm]])
    print("\nRun-to-run noise")
    show_table([results["noise"]] if results["noise"] else [])
    print("\nEvidence")
    print(json.dumps(results["evidence"], indent=2))
    print("\nEvidence for the true cause, when it's a candidate")
    show_table([{"cause": cause, **row}
                for cause, row in results["coverage"].items()])
    print("\nFull ToG cost per case")
    show_table([results["cost"]] if results["cost"] else [])
    look = results["findings_look"]
    if look:
        print("\nfindings_look: run-to-run noise (its repeats rerun the finding "
              "check too, while Full ToG's rerun only the final call)")
        show_table([look["noise"]] if look["noise"] else [])
        print("\nfindings_look: evidence")
        print(json.dumps(look["evidence"], indent=2))
        print("\nfindings_look: cost per case")
        show_table([look["cost"]] if look["cost"] else [])
    reasoning = results["reasoning"]
    if reasoning:
        show_reasoning(results["all"], results["comparisons"], reasoning)


def show_reasoning(arms, comparisons, reasoning):
    print("\nReasoning arms: how the steps use the KG items shown (main runs)")
    show_table([{"arm": arm, **row}
                for arm, row in reasoning["grounding"].items() if row])
    print("\nReasoning changes against reason_none, as the share of cases with"
          " at least one change and the mean steps per case. Noise compares "
          "repeat runs of the same arm; stable changes hold in every run of "
          "both arms; provable ones are stable and rest on a shown item they "
          "agree with or say doesn't apply, contradicting none (omissions: "
          "one a shown rule contradicts). "
          "Omissions involving findings can only be statistical.")
    for name, summary in reasoning["changes"].items():
        print(f"\n  {name}")
        show_table([{"row": row, **values}
                    for row, values in summary["rows"].items()
                    if values["cases"]])
        print(f"  Provable added or changed steps, by the kinds of items they "
              f"rest on: {summary['provable_by_kind'] or 'none'}")
    if reasoning["placebo_tracking"]:
        print("\nPlacebo steps citing a reversed rule or a moved finding: how "
              "many agree with the item shown and with the real one")
        show_table(reasoning["placebo_tracking"])
    difference = comparisons.get("reason_kg vs reason_none")
    kg = arms.get("reason_kg")
    if difference and kg:
        fails = difference["low"] > 0 or (kg["harms"] > kg["rescues"]
                                          and kg["mcnemar_p"] < 0.05)
        print(f"\nAccuracy guardrail for reason_kg: "
              f"{'FAILS' if fails else 'passes'} (log loss vs reason_none "
              f"{difference['difference']:+.3f}, 95% range "
              f"{difference['low']:+.3f} to {difference['high']:+.3f}; "
              f"{kg['rescues']} rescues and {kg['harms']} harms vs baseline, "
              f"p = {kg['mcnemar_p']:.3f})")


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
    runs_path = out / "arm_runs.json"
    arm_runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    for arm, arm_info in arm_runs.items():
        print(f"Arm {arm}: {arm_info}")
    if info.get("dirty") or any(i["dirty"] for i in arm_runs.values()):
        print("WARNING: these results came from uncommitted code.")
    print(f"Left out, awaiting a label check: {excluded or 'none'}")
    fingerprints = {json.dumps(arm_runs[arm].get("kg"), sort_keys=True)
                    for arm in REASON_ARMS if arm in arm_runs}
    reasoning = len(fingerprints) <= 1
    if not reasoning:
        print("WARNING: the reasoning arms used different KG files, so their "
              "reasoning measures are left out.")

    summary = {"run": info, "arm_runs": arm_runs, "excluded": excluded}
    for set_name in SETS:
        labels = {c["case_id"]: c["cause_name"] for c in sets[set_name]}
        groups = {"chain": [c for c in plan["chain"] if c in labels]}
        for key in ("cause_name", "age_group"):
            for case in sets[set_name]:
                groups.setdefault(case[key], []).append(case["case_id"])
        results = score_set(out, set_name, labels, groups, plan["lenient"],
                            reasoning)
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
