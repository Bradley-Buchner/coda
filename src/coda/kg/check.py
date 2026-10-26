"""Check an agent's top causes of death against the ACME rules in the KG."""
from coda.kg.query import KGQuery, normalize_code, verdict

# What it means for a cause to be listed in an ACME table, other than Table A
# Table A membership means “valid as an underlying cause”
TABLE_FLAGS = {
    "B": "can only be a contributing cause, not the underlying cause",
    "C1": "is not valid as an underlying cause in the US, but is under WHO "
          "rules",
    "C2": "can only be a contributing cause, not the underlying cause",
    "C3": "is a contributing cause with no effect on the underlying cause",
    "H": "is a trivial condition, which is not selected as the underlying "
         "cause if a more serious condition is reported",
}


def check_top_causes(kg: KGQuery, causes: dict) -> dict:
    """Return ACME flags on an agent's top causes and the links between them.

    Parameters
    ----------
    kg : KGQuery
        The ACME rules to check against.
    causes : dict
        The causes returned by an agent, keyed by CURIE, e.g. icd10:J18.9.
        Causes coded in other systems than ICD-10 are not checked.
    """
    nodes = {}
    flags = []
    for curie in causes:
        if not curie.startswith("icd10:"):
            continue
        node = kg.resolve(curie)
        if node is None:
            flags.append({"curie": curie, "code": normalize_code(curie),
                          "table": None, "flag": "is not in the ACME tables"})
        else:
            nodes[curie] = node
    for curie, node in nodes.items():
        for table in kg.validity(node):
            if table in TABLE_FLAGS:
                flags.append({"curie": curie, "code": node, "table": table,
                              "flag": TABLE_FLAGS[table]})
    links = []
    for cause, cause_node in nodes.items():
        for consequence, consequence_node in nodes.items():
            if cause == consequence:
                continue
            evidence = kg.due_to(cause_node, consequence_node)
            link_verdict = verdict(evidence)
            # Catch-all rules allow almost any link, so they say little
            if link_verdict in ("accepted", "needs_review"):
                links.append({"cause": cause, "consequence": consequence,
                              "verdict": link_verdict,
                              "table_d": evidence["table_d"],
                              "table_e": evidence["table_e"]})
    return {"flags": flags, "links": links}


def render_context(check: dict, causes: dict) -> str:
    """Return the result of check_top_causes as lines of text."""
    lines = []
    # Causes that cannot be the underlying cause
    contributing = {flag["curie"] for flag in check["flags"]
                    if flag["table"] in ("B", "C2", "C3")}
    for link in check["links"]:
        cause = causes[link["cause"]]["name"]
        consequence = causes[link["consequence"]]["name"]
        table = "Table E" if link["table_e"] else "Table D"
        line = f"- {consequence} may be due to {cause} (ACME {table}"
        if link["verdict"] == "needs_review":
            reasons = [rule["reason"] for rule in link["table_e"]
                       if rule["reason"]]
            line += f", only if: {reasons[0]}" if reasons else \
                ", only in some cases"
        line += ")."
        results = [rule["result"] for rule in link["table_e"]
                   if rule["result"]]
        if results:
            line += f" Together they are coded as {results[0]}."
        elif link["cause"] not in contributing:
            line += f" If both are reported, {cause} is the underlying cause."
        lines.append(line)
    for flag in check["flags"]:
        name = causes[flag["curie"]]["name"]
        lines.append(f"- {name} ({flag['code'][6:]}) {flag['flag']}.")
    return "\n".join(lines)
