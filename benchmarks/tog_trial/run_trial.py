"""Compare the backward Think-on-Graph agent with the prompted ICD-10 agent.

Runs ToGBackwardIcd10Agent on the seven English narratives that the
synthetic_narrative benchmark reads (lang_support's references), prints what
happened in each case, and saves the result and the trace of every LLM call
to results/<case_id>.json. The ToG agent's first opinion is the prompted
ICD-10 agent's answer, so the closing table compares the two on the same
call. These seven cases serve as a smoke test, not an evaluation.

--cited prints the facts and findings each saved result cited, and the
table, without running anything.

Usage
-----
    python benchmarks/tog_trial/run_trial.py --cases lri_1,malaria_1
    python benchmarks/tog_trial/run_trial.py
    python benchmarks/tog_trial/run_trial.py --cited
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from coda.config import settings
from coda.inference.tog import ToGBackwardIcd10Agent
from coda.kg.graph import KGView
from coda.llm_api import create_llm_client

CASES = (HERE.parent / "lang_support" / "data" / "bn" / "references"
         / "cases_bn_filtered.json")


def ranking(causes):
    return ", ".join(f"{c['name']} {c['score']:.2f}" for c in causes.values())


def top(causes):
    return max(causes.values(), key=lambda c: c["score"])["name"] \
        if causes else "-"


def compare(rows):
    """Print the label, each agent's top cause and the totals."""
    print(f"\n{'case':12} {'label':30} {'prompted agent':30} {'ToG':30} calls")
    for case_id, label, base, tog, calls in rows:
        print(f"{case_id:12} {label[:30]:30} "
              + " ".join(f"{name[:28]:28}{' ✓' if name == label else '  '}"
                         for name in (base, tog)) + f" {calls:5}")
    print(f"top-1 correct: prompted agent "
          f"{sum(r[2] == r[1] for r in rows)}/{len(rows)}, ToG "
          f"{sum(r[3] == r[1] for r in rows)}/{len(rows)}, "
          f"{sum(r[4] for r in rows)} calls in all")


def cited(evidence):
    """Print each cited fact or finding with the candidates it came from."""
    if not evidence["cited"]:
        print("    (nothing cited)")
    for item in evidence["cited"]:
        sources = []
        for candidate in evidence["candidates"]:
            if item in candidate["triples"]:
                sources.append(f"fact for {candidate['cause']}")
            sources += [f"{f['status']} finding, grade {f['grade']} for "
                        f"{candidate['cause']}"
                        for f in candidate["findings"] or []
                        if f["finding"] == item]
        print(f"    * {item}\n      ({'; '.join(sources)})")


def report(case_id, label, result, trace):
    """Print the first opinion, each candidate's evidence and the ranking."""
    evidence = result["kg_evidence"]
    print(f"\n{case_id} (written for {label}), {len(trace)} LLM calls\n"
          f"  first opinion: {ranking(trace[0]['response']['causes'])}")
    for candidate in evidence["candidates"]:
        findings = candidate["findings"]
        checked = "check failed" if findings is None else ", ".join(
            f"{sum(f['status'] == s for f in findings)} {s}"
            for s in ("present", "absent", "not_mentioned"))
        print(f"  {candidate['cause']}: findings {checked}")
        for triple in candidate["triples"]:
            print(f"    + {triple}")
    print(f"  final ranking: {ranking(result['causes'])}")
    cited(evidence)
    print(f"  reasoning: {result['reasoning']}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", help="Comma-separated case ids, such as "
                        "lri_1; default all")
    parser.add_argument("--provider", default=settings.inference.llm.provider)
    parser.add_argument("--model", default=settings.inference.llm.model)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--cited", action="store_true",
                        help="Show what the saved results in --out cited")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text())
    if args.cases:
        cases = [c for c in cases if c["case_id"] in args.cases.split(",")]
    out = Path(args.out)
    if args.cited:
        rows = []
        for case in cases:
            saved = json.loads((out / f"{case['case_id']}.json").read_text())
            result, trace = saved["result"], saved["trace"]
            print(f"\n{case['case_id']} (written for {saved['label']}): "
                  f"{ranking(result['causes'])}")
            cited(result["kg_evidence"])
            print(f"  reasoning: {result['reasoning']}")
            rows.append((case["case_id"], saved["label"],
                         top(trace[0]["response"]["causes"]),
                         top(result["causes"]), len(trace)))
        compare(rows)
        return
    client = create_llm_client(provider=args.provider, model=args.model)
    agent = ToGBackwardIcd10Agent(client, graph=KGView())
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        case_id, narrative = case["case_id"], case["narrative"]
        label = case["cause_name"]
        agent.reset()
        result = await agent.process_chunk(case_id, narrative, [])
        (out / f"{case_id}.json").write_text(json.dumps(
            {"label": label, "narrative": narrative, "result": result,
             "trace": agent.trace}, indent=2))
        report(case_id, label, result, agent.trace)
        rows.append((case_id, label, top(agent.trace[0]["response"]["causes"]),
                     top(result["causes"]), len(agent.trace)))
    compare(rows)


if __name__ == "__main__":
    asyncio.run(main())
