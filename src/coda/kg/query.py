"""Look up the ACME cause-of-death rules in the CODA KG files."""
import pandas as pd

from coda.kg.sources import KG_BASE

# Table E rule types for when the tentative underlying cause is a 'direct sequel'
# of the other condition, and the ones for when it is reported 'due to' the other
# condition. The other types (LMP, LMC, SENMC, SMP, SMC) only need both conditions
# to be reported.
SEQUEL_RULE_TYPES = {"DS", "DSC"}
DUE_TO_RULE_TYPES = {"IDDC", "SENDC", "LDP", "LDC", "SDC"}
# Table D rules that go through ranges of at least this many codes are
# deemed 'catch-all' rules
CATCH_ALL_SIZE = 1000


def normalize_code(code: str) -> str:
    """Return a KG node ID for an ICD-10 code, e.g. A419 -> icd10:A41.9."""
    code = code.replace("icd10:", "").replace(".", "").strip().upper()
    return f"icd10:{code}" if len(code) <= 3 else f"icd10:{code[:3]}.{code[3:]}"


class KGQuery:
    """Look up ACME rules in the files written by the ACME KG exporter."""

    def __init__(self, kg_dir=KG_BASE):
        edges = pd.read_csv(
            kg_dir.joinpath("acme_edges.tsv.gz"), sep="\t", dtype=str,
            keep_default_na=False,
        )
        # The ranges each code is part of, and the number of codes in each
        self.code_ranges: dict[str, set] = {}
        self.range_sizes: dict[str, int] = {}
        part_of_range = edges[edges[":TYPE"] == "part_of_range"]
        for code, code_range in zip(part_of_range[":START_ID"],
                                    part_of_range[":END_ID"]):
            self.code_ranges.setdefault(code, set()).add(code_range)
            self.range_sizes[code_range] = \
                self.range_sizes.get(code_range, 0) + 1
        # Table D: for each condition, the codes and ranges it may be due to
        self.causes: dict[str, dict] = {}
        causes = edges[edges[":TYPE"] == "causes"]
        for cause, consequence, maybe in zip(causes[":START_ID"],
                                             causes[":END_ID"],
                                             causes["maybe:boolean"]):
            self.causes.setdefault(consequence, {})[cause] = maybe == "True"
        # Table E: the modification rules of each tentative underlying cause
        self.rules: dict[str, list] = {}
        rules = edges[edges[":TYPE"] == "acme_modification"]
        for rule in rules.to_dict(orient="records"):
            self.rules.setdefault(rule[":END_ID"], []).append({
                "source": rule[":START_ID"],
                "rule_type": rule["rule_type"],
                "result": rule["result"],
                "maybe": rule["maybe:boolean"] == "True",
                "reason": rule["reason"],
            })
        # Tables A, B, C1-C3 and H: the tables that list each code
        self.code_lists: dict[str, list] = {}
        code_lists = edges[edges[":TYPE"] == "in_code_list"]
        for code, table in zip(code_lists[":START_ID"], code_lists[":END_ID"]):
            self.code_lists.setdefault(code, []).append(
                table.replace("acme:table_", "")
            )

    def resolve(self, code: str) -> str | None:
        """Return the KG node that the ACME tables use for a code.

        A code that the tables do not use is mapped to its parent if the
        tables use that (e.g. C80.1 -> C80), else to its unspecified
        subcode (e.g. E14 -> E14.9). Returns None if neither is used.
        """
        node = normalize_code(code)
        candidates = [node, node[:9]] if "." in node else [node, node + ".9"]
        for candidate in candidates:
            if candidate in self.causes or candidate in self.code_lists:
                return candidate
        return None

    def validity(self, node: str) -> list[str]:
        """Return the ACME tables (A, B, C1, C2, C3, H) that list a code."""
        return self.code_lists.get(node, [])

    def due_to(self, cause: str, consequence: str) -> dict:
        """Return the ACME rules saying that a consequence may be due to a cause.

        Returns the most specific Table D rule that allows it ('table_d', with
        the code or range it goes through, whether it is a "maybe" rule, and
        the number of codes in it), or None, and the Table E rules saying it.
        """
        cause_ranges = self.code_ranges.get(cause, set())
        table_d = None
        for start, maybe in self.causes.get(consequence, {}).items():
            if start == cause or start in cause_ranges:
                size = self.range_sizes.get(start, 1)
                if table_d is None or size < table_d["size"]:
                    table_d = {"via": start, "maybe": maybe, "size": size}
        table_e = []
        for rule in self.rules.get(consequence, []):
            if rule["rule_type"] in SEQUEL_RULE_TYPES and \
                    (rule["source"] == cause or rule["source"] in cause_ranges):
                table_e.append(rule)
        consequence_ranges = self.code_ranges.get(consequence, set())
        for rule in self.rules.get(cause, []):
            if rule["rule_type"] in DUE_TO_RULE_TYPES and \
                    (rule["source"] == consequence
                     or rule["source"] in consequence_ranges):
                table_e.append(rule)
        return {"table_d": table_d, "table_e": table_e}


def verdict(evidence: dict) -> str:
    """Return how the ACME rules treat a 'due to' link.

    Is 'accepted' if a specific rule allows it, 'needs_review' if only rules
    marked "maybe" do, 'accepted_catch_all' if only a catch-all Table D rule
    does, and 'not_listed' otherwise.
    """
    table_d = evidence["table_d"]
    if any(not rule["maybe"] for rule in evidence["table_e"]) or \
            (table_d and not table_d["maybe"]
             and table_d["size"] < CATCH_ALL_SIZE):
        return "accepted"
    if evidence["table_e"] or (table_d and table_d["maybe"]):
        return "needs_review"
    if table_d:
        return "accepted_catch_all"
    return "not_listed"
