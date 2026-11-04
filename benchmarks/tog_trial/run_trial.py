"""Smoke test of the backward Think-on-Graph agent on synthetic narratives.

Runs ToGBackwardIcd10Agent on the narratives of ../hw_benchmark, prints what
happened in each case, and saves the result and the trace of every LLM call
to results/<case_id>.json. This is not an evaluation.

Usage
-----
    python benchmarks/tog_trial/run_trial.py --cases 1,6
    python benchmarks/tog_trial/run_trial.py
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

NARRATIVES = HERE.parent / "hw_benchmark" / "narratives.json"

# CHAMPS cause each narrative was written for, in order
LABELS = ["Lower respiratory infections", "Diarrheal Diseases",
          "Neonatal sepsis", "Malaria", "Upper respiratory infections",
          "Measles", "Meningitis/Encephalitis", "Malnutrition",
          "Congenital birth defects or Heart Diseases", "Injury"]


def ranking(causes):
    return ", ".join(f"{c['name']} {c['score']:.2f}" for c in causes.values())


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
    for item in evidence["cited"]:
        print(f"    * {item}")
    print(f"  reasoning: {result['reasoning']}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", help="Comma-separated case numbers "
                        "(1-10); default all")
    parser.add_argument("--provider", default=settings.inference.llm.provider)
    parser.add_argument("--model", default=settings.inference.llm.model)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    narratives = json.loads(NARRATIVES.read_text())
    numbers = [int(n) for n in args.cases.split(",")] if args.cases \
        else range(1, len(narratives) + 1)
    client = create_llm_client(provider=args.provider, model=args.model)
    agent = ToGBackwardIcd10Agent(client, graph=KGView())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        case_id = f"case_{number:02d}"
        narrative, label = narratives[number - 1], LABELS[number - 1]
        agent.reset()
        result = await agent.process_chunk(case_id, narrative, [])
        (out / f"{case_id}.json").write_text(json.dumps(
            {"label": label, "narrative": narrative, "result": result,
             "trace": agent.trace}, indent=2))
        report(case_id, label, result, agent.trace)


if __name__ == "__main__":
    asyncio.run(main())
