"""Check an agent's top causes of death, or the 'due to' chain of an MCCD,
against the ACME rules in the KG."""
from coda.kg.query import (
    DUE_TO_RULE_TYPES,
    SEQUEL_RULE_TYPES,
    KGQuery,
    normalize_code,
    verdict,
)

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


def modify(kg: KGQuery, tentative: str, position: int, nodes: list,
           used: set) -> tuple:
    """Return the Table E rules that apply to a tentative underlying cause.

    The tentative cause is on line `position` of Part 1, whose codes are
    `nodes` from line (a) down––the lines in `used` were already used by a
    modification. Returns the first applicable rule not marked "maybe", or
    None with the line of its other condition, and the applicable rules
    marked "maybe" with the lines of their other conditions.
    """
    maybe_rules = []
    for rule in kg.rules.get(tentative, []):
        for other_position, other in enumerate(nodes):
            if other_position == position or other_position in used or not (
                    rule["source"] == other
                    or rule["source"] in kg.code_ranges.get(other, set())):
                continue
            # DS and DSC need the other condition on a lower line, and the
            # 'due to' types need the tentative cause on a lower line
            if rule["rule_type"] in SEQUEL_RULE_TYPES and \
                    other_position < position:
                continue
            if rule["rule_type"] in DUE_TO_RULE_TYPES and \
                    other_position > position:
                continue
            if rule["maybe"]:
                maybe_rules.append((rule, other_position))
            else:
                return rule, other_position, maybe_rules
    return None, None, maybe_rules


def check_mccd_chain(kg: KGQuery, codes: list) -> dict:
    """Return the ACME check of the 'due to' chain in Part 1 of an MCCD.

    Parameters
    ----------
    kg : KGQuery
        The ACME rules to check against.
    codes : list
        The ICD-10 codes of Part 1, from line (a) down, with empty strings
        for unused lines.

    Returns the verdict of each link between used lines, and the underlying
    cause suggested by the General Principle, Rule 1 or Rule 2 of the ACME
    Table D rules, then modified by the Table E rules not marked "maybe".
    """
    lines = [(chr(97 + i), code) for i, code in enumerate(codes) if code]
    nodes = [kg.resolve(code) for _, code in lines]
    links = []
    for i in range(1, len(lines)):
        verdict_i = "not_in_acme_tables" if None in (nodes[i], nodes[i - 1]) \
            else verdict(kg.due_to(nodes[i], nodes[i - 1]))
        links.append({"upper": lines[i - 1], "lower": lines[i],
                      "verdict": verdict_i})
    nodes = [node for node in nodes if node]
    if not nodes:
        return {"links": links, "underlying": None, "steps": []}
    # General Principle: the lowest line that could've given rise to everything above
    if all(verdict(kg.due_to(nodes[-1], upper)) != "not_listed"
           for upper in nodes[:-1]):
        position, steps = len(nodes) - 1, ["General Principle"]
    else:
        # Rule 1: the start of the sequence of links ending at line (a), or
        # Rule 2: the first condition mentioned
        position = 0
        while position + 1 < len(nodes) and verdict(
                kg.due_to(nodes[position + 1], nodes[position])) != "not_listed":
            position += 1
        steps = ["Rule 1" if position else "Rule 2"]
    tentative = nodes[position]
    steps[0] += f" selects {tentative[6:]}"
    # Apply Table E modifications
    used, review = set(), []
    for _ in range(5):
        rule, other_position, maybe_rules = modify(kg, tentative, position,
                                                   nodes, used)
        review += [f"Table E {maybe_rule['rule_type']} would select "
                   f"{maybe_rule['result'] or nodes[maybe_position][6:]} "
                   f"only if: {maybe_rule['reason'] or 'in some cases'}"
                   for maybe_rule, maybe_position in maybe_rules]
        if rule is None:
            break
        used.update({position, other_position})
        if rule["result"]:
            tentative = normalize_code(rule["result"])
        else:
            tentative, position = nodes[other_position], other_position
        steps.append(f"Table E {rule['rule_type']} selects {tentative[6:]}")
    return {"links": links, "underlying": tentative[6:], "steps": steps,
            "review": review,
            "flags": [TABLE_FLAGS[table] for table in kg.validity(tentative)
                      if table in TABLE_FLAGS]}


# How each link verdict reads on the MCCD
LINK_VERDICTS = {
    "accepted": "accepted by a specific ACME rule",
    "needs_review": "accepted only in some cases",
    "accepted_catch_all": "accepted by a broad ACME rule",
    "not_listed": "not accepted by the ACME rules",
    "not_in_acme_tables": "a code is not in the ACME tables",
}


def render_mccd_check(check: dict) -> str:
    """Return the result of check_mccd_chain as lines of text."""
    lines = []
    for link in check["links"]:
        (upper, upper_code), (lower, lower_code) = link["upper"], link["lower"]
        lines.append(f"- ({upper}) {upper_code} due to ({lower}) {lower_code}: "
                     f"{LINK_VERDICTS[link['verdict']]}.")
    if check["underlying"]:
        lines.append(f"Suggested underlying cause: {check['underlying']} "
                     f"({'; '.join(check['steps'])}). A suggestion, not "
                     f"official coding.")
    for review in check.get("review", []):
        lines.append(f"Needs review: {review}.")
    for flag in check.get("flags", []):
        lines.append(f"{check['underlying']} {flag}.")
    return "\n".join(lines) or "No ICD-10 codes in the ACME tables in Part 1."

