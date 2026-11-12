"""Run the arms of the benchmark on whether Think-on-Graph is worth adopting.

Saves one result per set, arm and case to <out>/<set>/<arm>/<case_id>.json, in
the format of run_trial.py. The full arm runs ToGBackwardIcd10Agent. The others
repeat only its final call on the same first opinion, with the evidence changed:

    second_look    no evidence
    findings_only  findings, no facts
    facts_only     facts, no findings
    placebo        each candidate gets the next candidate's evidence, relabeled

The findings_look arm runs ToGFindingsIcd10Agent's own calls on the same first
opinion: a new finding check and a final call that sees only findings.

The pilot and original sets also repeat the first opinion (first_rep*), the
full arm's final call (full_rep*) and the findings_look arm (findings_look_rep*),
to measure run-to-run noise.

A rerun skips saved results, so it can resume. Results with a failed LLM call are
not saved: they are logged to <out>/failed_calls.jsonl and retried on a rerun. A
run refuses to continue with other code or another model, or with uncommitted
changes unless --allow-dirty is given. --arms runs only the arms named, with
their repeats, on the saved full results, and records each arm's code and model
in <out>/arm_runs.json, so arms can be added to a finished run.

Usage
-----
    python benchmarks/tog_trial/run_benchmark.py --sets pilot
    python benchmarks/tog_trial/run_benchmark.py --sets original,sparse
    python benchmarks/tog_trial/run_benchmark.py --sets original,sparse \
        --arms findings_look
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coda.config import settings
from coda.inference.champs_prompted.icd10_agent import ChampsPromptedIcd10Agent
from coda.inference.tog import ToGBackwardIcd10Agent, ToGFindingsIcd10Agent
from coda.kg.graph import KGView
from coda.llm_api import create_llm_client

SETS = ("pilot", "original", "sparse")
ARMS = ("full", "second_look", "findings_only", "facts_only", "placebo",
        "findings_look", "first")  # first: the first-opinion repeats
REPEATED_SETS = ("pilot", "original")
REPEATS = 2
RUN_FILES = ("run_benchmark.py", "cases.json", "cases_sparse.json",
             "chain_cases.json")


def load_sets():
    """Return each set's cases: the pilot cases, the others, and their sparse
    versions.
    """
    cases = json.loads((HERE / "cases.json").read_text())
    pilot = json.loads((HERE / "chain_cases.json").read_text())["pilot"]
    return {
        "pilot": [c for c in cases if c["case_id"] in pilot],
        "original": [c for c in cases if c["case_id"] not in pilot],
        "sparse": json.loads((HERE / "cases_sparse.json").read_text()),
    }


def rotated(graph, candidates):
    """Return the candidates with each one holding the next one's evidence,
    relabeled as its own, or None if there is only one candidate.
    """
    if len(candidates) < 2:
        return None

    def start_name(candidate):
        start = graph.resolve(candidate["code"])
        return graph.name(start) if start else candidate["code"]

    placebo = []
    for i, candidate in enumerate(candidates):
        source = candidates[(i + 1) % len(candidates)]
        findings = source["findings"]
        placebo.append({
            **candidate,
            "triples": [triple.replace(start_name(source),
                                       start_name(candidate))
                        for triple in source["triples"]],
            "findings": findings and [{**f, "who_va_cause": candidate["cause"]}
                                      for f in findings],
        })
    return placebo


def control_arms(graph, candidates):
    """Return the candidates each control arm sees; None skips the arm."""
    return {
        "second_look": [{**c, "same_as": "", "triples": [], "findings": []}
                        for c in candidates],
        "findings_only": [{**c, "same_as": "", "triples": []}
                          for c in candidates],
        "facts_only": [{**c, "findings": []} for c in candidates],
        "placebo": rotated(graph, candidates),
    }


def failed_steps(agent, trace):
    """Return the steps of a trace whose LLM call gave no usable answer."""
    steps = [e["step"] for e in trace if e["response"] is None]
    for entry in trace:
        if entry["response"] is None:
            continue
        if entry["step"] == "first_opinion" \
                and not entry["response"]["causes"]:
            steps.append(entry["step"])
        if entry["step"] == "final_ranking" and not any(
                agent._match_cause(c["cause_name"])
                for c in entry["response"].get("top_causes", [])):
            steps.append(entry["step"])
    return steps


def run_info(provider, model, allow_dirty):
    """Return what identifies this run, or exit if it can't be trusted."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()

    tracked = git("ls-files", "benchmarks/tog_trial").splitlines()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no")) \
        or bool(git("ls-files", "--others", "--exclude-standard", "src")) \
        or any(f"benchmarks/tog_trial/{name}" not in tracked
               for name in RUN_FILES)
    if dirty and not allow_dirty:
        sys.exit("Commit your changes first, so that the results record the "
                 "code that made them.")
    return {"commit": git("rev-parse", "HEAD"), "dirty": dirty,
            "provider": provider, "model": model}


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", default="pilot",
                        help=f"Comma-separated, from {', '.join(SETS)}")
    parser.add_argument("--cases", help="Comma-separated case ids; default all")
    parser.add_argument("--arms", help="Comma-separated arms to run on the saved "
                        f"full results, with their repeats, from {', '.join(ARMS)}; "
                        "default all")
    parser.add_argument("--provider", default=settings.inference.llm.provider)
    parser.add_argument("--model", default=settings.inference.llm.model)
    parser.add_argument("--out", default=str(HERE / "results" / "benchmark"))
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Allow uncommitted changes, for debugging only")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = run_info(args.provider, args.model, args.allow_dirty)
    arms = args.arms.split(",") if args.arms else None
    if arms:
        unknown = set(arms) - set(ARMS)
        if unknown:
            sys.exit(f"Unknown arms: {', '.join(sorted(unknown))}")
        runs_path = out / "arm_runs.json"
        arm_runs = json.loads(runs_path.read_text()) \
            if runs_path.exists() else {}

        def saved(arm):
            return any(folder.name == arm or folder.name.startswith(f"{arm}_rep")
                       for set_name in SETS if (out / set_name).exists()
                       for folder in (out / set_name).iterdir()
                       if any(folder.glob("*.json")))

        clashes = [arm for arm in arms
                   if arm_runs.get(arm, info) != info and saved(arm)]
        if clashes:
            sys.exit(f"{out} holds {', '.join(clashes)} results from other code "
                     "or another model. Use a new --out.")
        runs_path.write_text(json.dumps(
            {**arm_runs, **{arm: info for arm in arms}}, indent=2))
    else:
        info_path = out / "run_info.json"
        if info_path.exists() and json.loads(info_path.read_text()) != info:
            sys.exit(f"{out} holds results from other code or another model. "
                     "Use a new --out.")
        info_path.write_text(json.dumps(info, indent=2))

    client = create_llm_client(provider=args.provider, model=args.model)
    graph = KGView()
    agent = ToGBackwardIcd10Agent(client, graph=graph)
    findings_agent = ToGFindingsIcd10Agent(client, graph=graph)
    base = ChampsPromptedIcd10Agent(client)
    sets = load_sets()

    async def saved_run(set_name, arm, case, run):
        """Return the saved result of an arm, running it if needed; None if a
        call failed.
        """
        path = out / set_name / arm / f"{case['case_id']}.json"
        if path.exists():
            return json.loads(path.read_text())
        result, trace = await run()
        steps = failed_steps(agent, trace)
        if steps:
            print(f"  {set_name}/{arm}/{case['case_id']}: failed at {steps}")
            with (out / "failed_calls.jsonl").open("a") as log:
                log.write(json.dumps(
                    {"set": set_name, "arm": arm, "case_id": case["case_id"],
                     "steps": steps}) + "\n")
            return None
        saved = {"label": case["cause_name"], "narrative": case["narrative"],
                 "result": result, "trace": trace}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(path)
        return saved

    async def run_full(case):
        agent.reset()
        result = await agent.process_chunk(case["case_id"], case["narrative"],
                                           [])
        return result, agent.trace

    async def run_first(case):
        base.reset()
        result = await base.process_chunk(case["case_id"], case["narrative"],
                                          [])
        return result, [{"step": "first_opinion", "response": result}]

    def case_input(full):
        final = next(e for e in full["trace"] if e["step"] == "final_ranking")
        return final["user_prompt"].split("\n\n## CANDIDATES")[0]

    async def replay(case, full, candidates):
        """Repeat the full arm's final call with these candidates."""
        agent.trace = []
        result = await agent.rank(case["case_id"],
                                  full["trace"][0]["response"],
                                  case_input(full), candidates)
        return result, agent.trace

    async def run_findings(case, full):
        """Run the findings-only agent's calls on the full arm's first opinion."""
        findings_agent.trace = []
        started = time.perf_counter()
        result = await findings_agent.rerank(
            case["case_id"], full["trace"][0]["response"], case_input(full))
        result["timings"] = {"inference_s": time.perf_counter() - started}
        return result, findings_agent.trace

    def wanted(arm):
        return not arms or any(arm == name or arm.startswith(f"{name}_rep")
                               for name in arms)

    chosen = args.cases.split(",") if args.cases else None
    for set_name in args.sets.split(","):
        cases = [c for c in sets[set_name]
                 if not chosen or c["case_id"] in chosen]
        for number, case in enumerate(cases, 1):
            print(f"{set_name} {number}/{len(cases)}: {case['case_id']}")
            full_path = out / set_name / "full" / f"{case['case_id']}.json"
            if not wanted("full") and not full_path.exists():
                print("  No saved full result, so skipped")
                continue
            full = await saved_run(set_name, "full", case,
                                   lambda: run_full(case))
            if full is None:
                continue
            candidates = full["result"]["kg_evidence"]["candidates"]
            runs = {arm: (lambda arm_candidates=arm_candidates:
                          replay(case, full, arm_candidates))
                    for arm, arm_candidates
                    in control_arms(graph, candidates).items()
                    if arm_candidates is not None}
            runs["findings_look"] = lambda: run_findings(case, full)
            if set_name in REPEATED_SETS:
                for i in range(1, REPEATS + 1):
                    runs[f"first_rep{i}"] = lambda: run_first(case)
                    runs[f"full_rep{i}"] = lambda: replay(case, full,
                                                          candidates)
                    runs[f"findings_look_rep{i}"] = \
                        lambda: run_findings(case, full)
            for arm, run in runs.items():
                if wanted(arm):
                    await saved_run(set_name, arm, case, run)
    print(f"Saved to {out}. Score with score_benchmark.py.")


if __name__ == "__main__":
    asyncio.run(main())
