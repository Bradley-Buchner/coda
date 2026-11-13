"""Numbered KG evidence for ToG's reasoning, and checks of the reasoning steps
that cite it.

The evidence is R items (ACME rules on how the candidates relate), F items
(typical findings the case shows) and W items (walk facts). A reasoning step
names the candidates it favors and disfavors, the underlying cause it implies,
and the items it rests on. The checks compare a step with the items it cites,
and the reasoning of a KG arm with that of an arm without KG evidence.
"""

from coda.inference.tog.backward import shown_findings
from coda.kg.check import render_context

ROLES = ["sequence", "coding_rule", "finding_fit", "tells_apart",
         "not_applicable", "other"]
# ACME tables whose causes are only contributing causes, and the table of
# trivial conditions, which give way to a more serious one
CONTRIBUTING_TABLES = ("B", "C2", "C3")
TRIVIAL_TABLE = "H"
# Checks of a step that rest it on the items it cites: it agrees with them, or
# says why they don't apply to the case
GROUNDED = ("consistent", "rejected")
NO_EVIDENCE = "(No KG evidence is shown.)"


def kg_items(check, causes, candidates, walks=False):
    """Return the numbered KG items for a case: R items from check_top_causes on
    the first opinion's causes, F items from the candidates' findings, and W
    items from their walk facts if walks is True.

    Flags on injury codes (S, T) are left out: ACME doesn't select them because
    it selects the external cause instead, which the Injury group includes.
    """
    check = {**check, "flags": [flag for flag in check["flags"]
                                if not flag["code"][6:].startswith(("S", "T"))]}
    items = []
    lines = render_context(check, causes).splitlines()
    for entry, line in zip(check["links"] + check["flags"], lines):
        item = {"kind": "R", "text": line.removeprefix("- ")}
        if "flag" in entry:
            item.update(candidate=causes[entry["curie"]]["name"],
                        table=entry["table"])
        else:
            item.update(cause=causes[entry["cause"]]["name"],
                        consequence=causes[entry["consequence"]]["name"],
                        both_ways="reverse" in entry)
        items.append(item)
    findings = {}
    for candidate in candidates:
        for f in shown_findings(candidate["findings"]):
            findings.setdefault((f["finding"], f["status"]), []).append(
                (candidate["cause"], f))
    for (finding, status), typical in findings.items():
        items.append({"kind": "F", "typical": [cause for cause, _ in typical],
                      "text": f"[{status}] {finding} Typical of " + "; ".join(
                          f"{cause} (grade {f['grade']} for {f['who_va_cause']}"
                          f" on answer {f['graded_answer']})"
                          for cause, f in typical)})
    if walks:
        triples = {}
        for candidate in candidates:
            for triple in candidate["triples"]:
                triples.setdefault(triple, candidate["cause"])
        items += [{"kind": "W", "text": triple, "candidate": cause}
                  for triple, cause in triples.items()]
    counts = {}
    for item in items:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
        item["id"] = f"{item['kind']}{counts[item['kind']]}"
    return items


def render_items(items):
    """Return the KG evidence block of a final ranking prompt."""
    return "## KG EVIDENCE\n" + ("\n".join(f"{item['id']}. {item['text']}"
                                           for item in items) or NO_EVIDENCE)


def step_names(step):
    return {*step["favors"], *step["disfavors"], step["underlying"]} - {""}


def item_verdict(item, step):
    """Return whether a step is consistent with one item it cites, rejects it
    as not applying to the case, contradicts it, or can't be checked by it.
    """
    names, favors = step_names(step), set(step["favors"])
    disfavors = set(step["disfavors"])
    if step["role"] == "not_applicable" and item["kind"] in ("R", "F"):
        return "rejected"
    if item["kind"] == "W":
        return "consistent" if item["candidate"] in names else "unchecked"
    if item["kind"] == "F":
        typical = set(item["typical"])
        # A typical finding used only against causes it is typical of; one
        # also typical of a favored cause can't tell them apart
        if disfavors & typical:
            return "unchecked" if favors & typical else "inconsistent"
        if favors & typical or (step["role"] == "tells_apart"
                                and len(names & typical) >= 2):
            return "consistent"
        return "unchecked"
    if "candidate" in item:
        candidate = item["candidate"]
        ruled_out = candidate in favors or candidate == step["underlying"]
        # A trivial condition gives way only to another condition in play
        if ruled_out and (item["table"] in CONTRIBUTING_TABLES or (
                item["table"] == TRIVIAL_TABLE
                and names - {candidate} - disfavors)):
            return "inconsistent"
        return "consistent" if candidate in names else "unchecked"
    # A one-way rule makes its cause the underlying cause when both are present
    if not item["both_ways"] and step["underlying"] == item["consequence"] \
            and item["cause"] in names and item["cause"] not in disfavors:
        return "inconsistent"
    if {item["cause"], item["consequence"]} <= names and (
            item["both_ways"] or step["underlying"] in (item["cause"], "")):
        return "consistent"
    return "unchecked"


def check_step(step, items):
    """Return how a step relates to the shown items it cites: inconsistent if
    any item contradicts it, else consistent if one supports it, else rejected
    if it says why one doesn't apply, else unchecked.
    """
    by_id = {item["id"]: item for item in items}
    verdicts = {item_verdict(by_id[i], step) for i in step["basis"]
                if i in by_id}
    return next((verdict for verdict in ("inconsistent", "consistent",
                                         "rejected") if verdict in verdicts),
                "unchecked")


def step_key(step):
    return step["role"], frozenset(step_names(step))


def step_direction(step):
    return (frozenset(step["favors"]), frozenset(step["disfavors"]),
            step["underlying"])


def pooled(steps):
    """Return {key: directions} over the steps of one run."""
    pool = {}
    for step in steps:
        pool.setdefault(step_key(step), set()).add(step_direction(step))
    return pool


def diff_steps(kg_steps, control_steps):
    """Return the step keys of a KG run that differ from a control run: added,
    omitted, changed (no direction in common), and grounded (same logic, now
    resting on a KG item it agrees with or says doesn't apply).
    """
    kg, control = pooled(kg_steps), pooled(control_steps)
    shared = kg.keys() & control.keys()
    return {
        "added": [k for k in kg if k not in control],
        "omitted": [k for k in control if k not in kg],
        "changed": [k for k in shared if not kg[k] & control[k]],
        "grounded": [k for k in shared if kg[k] & control[k] and any(
            step_key(s) == k and s["check"] in GROUNDED for s in kg_steps)],
    }


def stable_changes(kg_runs, control_runs, items):
    """Return the changes between repeat runs of a KG arm and of its control
    that hold in every run, and those of them that are provable, with the kinds
    of items each provable change rests on.

    A change is stable if its key is in every KG run and no control run
    (added), in every control run and no KG run (omitted), or in every KG run
    with a direction no control run gives (changed). An added or changed step
    is provable if, in every KG run, a step with that key (and, if changed,
    that direction) agrees with a shown item it cites or says why it doesn't
    apply, and contradicts none. An omitted step is provable if a
    shown R item contradicts it; findings can't, as in InterVA other answers
    aren't evidence against a cause.
    """
    kg, control = [pooled(r) for r in kg_runs], [pooled(r) for r in control_runs]
    in_all = set.intersection(*(set(p) for p in kg))
    changes = {"added": [], "omitted": [], "changed": []}
    common = {}
    for key in in_all:
        common[key] = set.intersection(*(p[key] for p in kg))
        given = set().union(*(p.get(key, set()) for p in control))
        if not given:
            changes["added"].append(key)
        elif common[key] and not common[key] & given:
            changes["changed"].append(key)
    changes["omitted"] = [k for k in set.intersection(*(set(p) for p in control))
                          if not any(k in p for p in kg)]

    def grounded(run, key, kind):
        return [s for s in run if step_key(s) == key
                and (kind == "added" or step_direction(s) in common[key])
                and s["check"] in GROUNDED]

    rules = [item for item in items if item["kind"] == "R"]
    by_id = {item["id"]: item for item in items}
    kinds = {}
    for kind in ("added", "changed"):
        changes[f"provable_{kind}"] = [
            k for k in changes[kind]
            if all(grounded(r, k, kind) for r in kg_runs)]
        for key in changes[f"provable_{kind}"]:
            kinds[key] = sorted({i[0] for s in grounded(kg_runs[0], key, kind)
                                 for i in s["valid"]
                                 if item_verdict(by_id[i], s) in GROUNDED})
    changes["provable_omitted"] = [
        k for k in changes["omitted"] if any(
            step_key(s) == k and check_step({**s, "basis": [r["id"]]}, rules)
            == "inconsistent" for r in rules for run in control_runs
            for s in run)]
    changes["kinds"] = kinds
    return changes
