"""Write cases_sparse.json: the test cases with half of their sentences removed.

Each case keeps its first sentence and a random selection of the others, fixed
by SEED and the case id, so that rerunning this script gives the same file. The
pilot cases are left out.

Usage
-----
    python benchmarks/tog_trial/make_sparse.py
"""
import json
import random
import re
from math import ceil
from pathlib import Path

HERE = Path(__file__).parent
SEED = 20261106


def sparse_narrative(case_id, narrative):
    """Return the narrative with half of its sentences removed at random,
    keeping the first sentence and the original order.
    """
    sentences = re.split(r"(?<=[.!?])\s+", narrative.strip())
    rng = random.Random(f"{SEED}:{case_id}")
    kept = rng.sample(range(1, len(sentences)), ceil(len(sentences) / 2) - 1)
    return " ".join(sentences[i] for i in sorted([0, *kept]))


def main():
    cases = json.loads((HERE / "cases.json").read_text())
    pilot = json.loads((HERE / "chain_cases.json").read_text())["pilot"]
    sparse = [{**case, "narrative": sparse_narrative(case["case_id"],
                                                     case["narrative"])}
              for case in cases if case["case_id"] not in pilot]
    (HERE / "cases_sparse.json").write_text(json.dumps(sparse, indent=2) + "\n")
    print(f"Wrote {len(sparse)} cases to cases_sparse.json")


if __name__ == "__main__":
    main()
