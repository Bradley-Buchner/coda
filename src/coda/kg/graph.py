"""In-memory view of the KG parts that Think-on-Graph walks use.

Reads the ICD-10 hierarchy, matching ICD-11 names, ACME causal edges,
WHO VA cause mappings and ProbBase likelihoods from the exported TSV
files in KG_BASE, so it needs the repository's kg/ folder (the Docker
images query the KG in Neo4j instead).
"""

import ast
from collections import defaultdict
from pathlib import Path

import pandas as pd

from coda.kg.sources import KG_BASE

# Relations a walk may follow. The ICD-11 names and the WHO VA map are fixed
# lookups.
WALK_RELATIONS = ("is_a", "causes")
DIRECTIONS = ("out", "in")
# ProbBase likelihood grades, most to least likely
GRADES = ["I", "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
          "D+", "D", "D-", "E", "N"]
# Weakest grade that counts as a strong finding
STRONG_GRADE = "A-"
# ACME sources that say nothing a case can use: senility, lab and imaging
# findings (R70-R94) and ill-defined deaths (R95-R99)
DROPPED_SOURCES = ("R54", "R7", "R8", "R9")
HUB_TARGETS = 300  # threshold defining a hub node


def read_tsv(path, columns):
    """Read the given columns of a KG TSV file as strings."""
    return pd.read_csv(path, sep="\t", usecols=columns, dtype=str) \
        .fillna("")


class KGView:
    """Lookups over the ICD-10/11, ACME, WHO VA and ProbBase parts of the KG."""

    def __init__(self, kg_base: Path = KG_BASE):
        self.names = {}
        # adjacency[relation][direction][node] -> set of neighbor nodes
        self.adjacency = {rel: {d: defaultdict(set) for d in DIRECTIONS}
                          for rel in WALK_RELATIONS}
        self.range_names = {}
        self.ranges_of = defaultdict(set)
        self.icd11_matches = {}
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
        self.load_icd11(kg_base)

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
        """Load ACME causes edges from sources that can inform a case.

        A range stays one node, named after the ICD-10 group holding it. Ranges
        wider than one ICD-10 block, such as A00.0-R82.5 causes J18.9, say
        little about any one cause and are dropped, as are DROPPED_SOURCES and
        edges from a source to itself or a code in its range. This leaves
        end-of-chain causes such as J18.9 without incoming ACME edges.
        """
        edges = read_tsv(path, [":START_ID", ":END_ID", ":TYPE"])
        members = defaultdict(set)
        ranges = edges[edges[":TYPE"] == "part_of_range"]
        for code, code_range in zip(ranges[":START_ID"], ranges[":END_ID"]):
            members[code_range].add(code)
        causes = edges[edges[":TYPE"] == "causes"]
        kept = {source for source in causes[":START_ID"].unique()
                if self.keep_source(source)}
        for source in kept:
            if source.startswith("icd10:("):
                first, last = ast.literal_eval(source.removeprefix("icd10:"))
                group = self.names.get(self.common_ancestor(first, last), "")
                self.range_names[source] = f"{first}–{last} {group}".strip()
                for code in members[source]:
                    self.ranges_of[code].add(source)
        for source, target in zip(causes[":START_ID"], causes[":END_ID"]):
            if source in kept and target not in members.get(source, {source}):
                self.add_edge("causes", source, target)

    def load_icd11(self, kg_base):
        """Load the names of ICD-11 codes WHO's map matches one-to-one.

        WHO maps each ICD-11 code to its closest ICD-10 code, so an ICD-10 code
        reached from several ICD-11 codes is not the same as any of them.
        """
        nodes = read_tsv(kg_base / "icd11_nodes.tsv.gz",
                         ["id:ID", "code", "name"])
        names = {node: " ".join(f"ICD-11 {code} {name}".split())
                 for node, code, name in zip(nodes["id:ID"], nodes["code"],
                                             nodes["name"])}
        edges = read_tsv(kg_base / "icd11_edges.tsv.gz",
                         [":START_ID", ":END_ID", ":TYPE"])
        edges = edges[edges[":TYPE"] == "maps_to"]
        mapped = defaultdict(list)
        for source, target in zip(edges[":START_ID"], edges[":END_ID"]):
            mapped[target].append(source)
        self.icd11_matches = {target: names[sources[0]]
                              for target, sources in mapped.items()
                              if len(sources) == 1}

    def keep_source(self, source):
        """Return whether an ACME source's causes edges are worth keeping."""
        code = source.removeprefix("icd10:")
        first, last = ast.literal_eval(code) if code.startswith("(") \
            else (code, code)
        if first.startswith(DROPPED_SOURCES) \
                or last.startswith(DROPPED_SOURCES):
            return False
        block = self.top_block(first)
        return first[:3] == last[:3] \
            or (block is not None and block == self.top_block(last))

    def ancestors(self, code):
        """Return a code's node and those above it in the ICD-10 hierarchy."""
        chain, node = [], self.resolve(code)
        while node:
            chain.append(node)
            parents = self.adjacency["is_a"]["out"].get(node)
            node = min(parents) if parents else None
        return chain

    def top_block(self, code):
        """Return the highest ICD-10 block above a code, such as C00-C97."""
        blocks = [node for node in self.ancestors(code) if "-" in node]
        return blocks[-1] if blocks else None

    def same_as(self, code):
        """Return the name of the ICD-11 code matching a candidate's code."""
        return self.icd11_matches.get(self.resolve(code), "")

    def common_ancestor(self, first, last):
        """Return the lowest ICD-10 node above both codes, if any."""
        above = set(self.ancestors(last))
        return next((n for n in self.ancestors(first) if n in above), None)

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
        if node in self.range_names:
            return self.range_names[node]
        name = self.names.get(node, "")
        code = node.split(":", 1)[-1]
        if name and node.startswith("icd10:"):
            return f"{code} {name}"
        return name or code

    def relations(self, node):
        """Return the (relation, direction) pairs a node has neighbors on."""
        return [(rel, direction) for rel in WALK_RELATIONS
                for direction in DIRECTIONS
                if self.neighbors(node, rel, direction)]

    def neighbors(self, node, relation, direction):
        """Return a node's neighbors along one relation, most specific first.

        A code's causes out include those of its ranges, without those of hubs.
        """
        adjacency = self.adjacency[relation]
        if relation == "causes" and direction == "out":
            found = set()
            for source in {node} | self.ranges_of.get(node, set()):
                targets = adjacency["out"].get(source, ())
                if len(targets) < HUB_TARGETS:
                    found.update(targets)
            found.discard(node)
        else:
            found = adjacency[direction].get(node, ())
        other_side = adjacency["in" if direction == "out" else "out"]
        return sorted(found, key=lambda n: (len(other_side.get(n, ())), n))

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
