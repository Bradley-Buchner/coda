"""In-memory view of the KG parts that Think-on-Graph walks use.

Reads the ICD-10 hierarchy, ACME causal edges, WHO VA cause mappings and
ProbBase likelihoods from the exported TSV files in KG_BASE, so it needs the
repository's kg/ folder (the Docker images query the KG in Neo4j instead).
"""

import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd

from coda.kg.sources import KG_BASE

# Relations a walk may follow. maps_to (ICD-10 to WHO VA) is never walked.
WALK_RELATIONS = ("is_a", "causes")
DIRECTIONS = ("out", "in")
# ProbBase likelihood grades, most to least likely
GRADES = ["I", "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
          "D+", "D", "D-", "E", "N"]
# Weakest grade that counts as a strong finding
STRONG_GRADE = "A-"


def read_tsv(path, columns):
    """Read the given columns of a KG TSV file as strings."""
    return pd.read_csv(path, sep="\t", usecols=columns, dtype=str) \
        .fillna("")


class KGView:
    """Lookups over the ICD-10, ACME, WHO VA and ProbBase parts of the KG."""

    def __init__(self, kg_base: Path = KG_BASE):
        self.names = {}
        # adjacency[relation][direction][node] -> set of neighbor nodes
        self.adjacency = {rel: {d: defaultdict(set) for d in DIRECTIONS}
                          for rel in WALK_RELATIONS}
        self.who_va_map = defaultdict(set)
        self.grades = defaultdict(dict)

        for source in ["icd10", "who_va", "probbase"]:
            nodes = read_tsv(kg_base / f"{source}_nodes.tsv.gz",
                             ["id:ID", "name"])
            self.names.update(zip(nodes["id:ID"], nodes["name"]))

        icd10 = read_tsv(kg_base / "icd10_edges.tsv.gz",
                         [":START_ID", ":END_ID"])
        for child, parent in zip(icd10[":START_ID"], icd10[":END_ID"]):
            self.add_edge("is_a", child, parent)

        self.load_acme(kg_base / "acme_edges.tsv.gz")

        who_va = read_tsv(kg_base / "who_va_edges.tsv.gz",
                          [":START_ID", ":END_ID", ":TYPE"])
        who_va = who_va[who_va[":TYPE"] == "maps_to"]
        for code, cause in zip(who_va[":START_ID"], who_va[":END_ID"]):
            self.who_va_map[code].add(cause)

        probbase = read_tsv(kg_base / "probbase_edges.tsv.gz",
                            [":START_ID", ":END_ID", "value", ":TYPE"])
        probbase = probbase[(probbase[":TYPE"] == "probbase_rel")
                            & probbase["value"].isin(GRADES)]
        for question, cause, grade in zip(probbase[":START_ID"],
                                          probbase[":END_ID"],
                                          probbase["value"]):
            self.grades[cause][question] = grade

        cutoff = GRADES.index(STRONG_GRADE)
        self.strong_questions = {
            cause: {q for q, g in grades.items() if GRADES.index(g) <= cutoff}
            for cause, grades in self.grades.items()}
        # Questions strong for most causes (such as sex and season) don't
        # tell causes apart
        counts = defaultdict(int)
        for questions in self.strong_questions.values():
            for question in questions:
                counts[question] += 1
        self.uninformative_questions = {
            q for q, n in counts.items()
            if n > len(self.strong_questions) / 2}

    def add_edge(self, relation, source, target):
        self.adjacency[relation]["out"][source].add(target)
        self.adjacency[relation]["in"][target].add(source)

    def load_acme(self, path):
        """Load ACME causes edges between codes.

        Edges from ranges within one 3-character category are expanded to
        the range's codes, except the target itself. Wider ranges, such as A00.0-R82.5 causes J18.9, say
        little about any one cause and are dropped, as are edges from R-chapter
        codes, which act as hubs. This leaves end-of-chain causes such as
        J18.9 without ACME edges.
        """
        edges = read_tsv(path, [":START_ID", ":END_ID", ":TYPE"])
        members = defaultdict(set)
        ranges = edges[edges[":TYPE"] == "part_of_range"]
        for code, code_range in zip(ranges[":START_ID"], ranges[":END_ID"]):
            members[code_range].add(code)
        causes = edges[edges[":TYPE"] == "causes"]
        for source, target in zip(causes[":START_ID"], causes[":END_ID"]):
            if source.startswith("icd10:("):
                first, last = ast.literal_eval(source.removeprefix("icd10:"))
                codes = members[source] if first[:3] == last[:3] else ()
            else:
                first, codes = source.removeprefix("icd10:"), [source]
            if not first.startswith("R"):
                for code in codes:
                    if code != target:
                        self.add_edge("causes", code, target)

    def resolve(self, code):
        """Return the KG node of an ICD-10 code (A41.9 or icd10:A41.9).

        Falls back to the 3-character category (C80.1 -> C80) and returns None
        if neither is in the KG.
        """
        node = code if code.startswith("icd10:") else f"icd10:{code}"
        if node in self.names:
            return node
        category = node.split(".")[0]
        return category if category in self.names else None

    def name(self, node):
        """Return a readable name, prefixed with the code for ICD-10 nodes."""
        name = self.names.get(node, "")
        code = node.split(":", 1)[-1]
        if name and node.startswith("icd10:"):
            return f"{code} {name}"
        return name or code

    def relations(self, node):
        """Return the (relation, direction) pairs a node has neighbors on."""
        return [(rel, direction) for rel in WALK_RELATIONS
                for direction in DIRECTIONS
                if self.adjacency[rel][direction].get(node)]

    def neighbors(self, node, relation, direction):
        """Return a node's neighbors along one relation, sorted by ID."""
        return sorted(self.adjacency[relation][direction].get(node, ()))

    def who_va_causes(self, code):
        """Return the WHO VA causes of an ICD-10 code or its nearest ancestor.

        A fixed lookup for the finding check, not a walk step.
        """
        node = self.resolve(code)
        while node:
            if self.who_va_map.get(node):
                return sorted(self.who_va_map[node])
            parents = self.adjacency["is_a"]["out"].get(node)
            node = min(parents) if parents else None
        return []

    def strong_findings(self, cause):
        """Return (question, grade) for a cause's informative strong links."""
        questions = self.strong_questions.get(cause, set()) \
            - self.uninformative_questions
        return sorted((q, self.grades[cause][q]) for q in questions)
