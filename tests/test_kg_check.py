import pandas as pd

from coda.kg import query
from coda.kg.check import (
    check_mccd_chain,
    check_top_causes,
    render_context,
    render_mccd_check,
)
from coda.kg.query import KGQuery, verdict
from coda.kg.sources import write_tsv_gz

RANGE = "icd10:('A00.0', 'R82.5')"
EDGES = [
    # :START_ID, :END_ID, :TYPE, maybe, rule_type, result, reason
    ["icd10:B24", RANGE, "part_of_range", "", "", "", ""],
    ["icd10:D64.9", RANGE, "part_of_range", "", "", "", ""],
    [RANGE, "icd10:J18.9", "causes", "False", "", "", ""],
    ["icd10:S72.0", "icd10:J18.9", "causes", "True", "", "", ""],
    ["icd10:B24", "icd10:J18.9", "acme_modification", "False", "DSC", "B20.8",
     ""],
    ["icd10:I50.9", "icd10:I10", "acme_modification", "True", "LDC", "I11.0",
     "Must be hypertensive heart disease"],
    ["icd10:B24", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:C80", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:E14.9", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:I10", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:I50.9", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:J18.9", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:T14.9", "acme:table_B", "in_code_list", "", "", "", ""],
    ["icd10:E43", "icd10:J18.9", "causes", "False", "", "", ""],
    ["icd10:J18.9", "icd10:E43", "causes", "True", "", "", ""],
    ["icd10:J18.9", "icd10:D64.9", "causes", "False", "", "", ""],
    ["icd10:R99", "icd10:J18.9", "causes", "False", "", "", ""],
    ["icd10:R99", "acme:table_A", "in_code_list", "", "", "", ""],
    [RANGE, "icd10:E43", "causes", "True", "", "", ""],
    ["icd10:E43", "icd10:T14.9", "causes", "False", "", "", ""],
    ["icd10:T14.9", "icd10:E43", "causes", "False", "", "", ""],
]
NAMES = [["icd10:B20.8",
          "HIV disease resulting in other infectious and parasitic diseases"]]


def make_kg(tmp_path):
    edges = pd.DataFrame(EDGES, columns=[
        ":START_ID", ":END_ID", ":TYPE", "maybe:boolean", "rule_type",
        "result", "reason",
    ])
    write_tsv_gz(edges, tmp_path / "acme_edges.tsv.gz")
    write_tsv_gz(pd.DataFrame(NAMES, columns=["id:ID", "name"]),
                 tmp_path / "icd10_nodes.tsv.gz")
    return KGQuery(kg_dir=tmp_path)


def test_resolve(tmp_path):
    kg = make_kg(tmp_path)
    assert kg.resolve("icd10:J18.9") == "icd10:J18.9"
    # Codes the tables do not use map to their parent or unspecified subcode
    assert kg.resolve("C80.1") == "icd10:C80"
    assert kg.resolve("E14") == "icd10:E14.9"
    assert kg.resolve("Q99") is None


def test_due_to(tmp_path, monkeypatch):
    kg = make_kg(tmp_path)
    evidence = kg.due_to("icd10:B24", "icd10:J18.9")
    assert evidence["table_d"] == {"via": RANGE, "maybe": False, "size": 2}
    assert evidence["table_e"][0]["result"] == "B20.8"
    assert verdict(evidence) == "accepted"
    assert verdict(kg.due_to("icd10:S72.0", "icd10:J18.9")) == "needs_review"
    # In LDC rules the tentative underlying cause is reported due to the
    # other condition
    evidence = kg.due_to("icd10:I10", "icd10:I50.9")
    assert evidence["table_e"][0]["rule_type"] == "LDC"
    assert verdict(kg.due_to("icd10:I50.9", "icd10:I10")) == "not_listed"
    assert verdict(kg.due_to("icd10:D64.9", "icd10:J18.9")) == "accepted"
    monkeypatch.setattr(query, "CATCH_ALL_SIZE", 2)
    assert verdict(kg.due_to("icd10:D64.9", "icd10:J18.9")) == \
        "accepted_catch_all"


def test_check_top_causes(tmp_path):
    kg = make_kg(tmp_path)
    causes = {
        "icd10:B24": {"name": "HIV"},
        "icd10:J18.9": {"name": "Lower respiratory infections"},
        "icd10:T14.9": {"name": "Injury"},
        "icd11:1C62.Z": {"name": "HIV disease"},
        "icd10:Q99": {"name": "Other chromosome abnormalities"},
    }
    check = check_top_causes(kg, causes)
    assert [(link["cause"], link["consequence"], link["verdict"])
            for link in check["links"]] == \
        [("icd10:B24", "icd10:J18.9", "accepted")]
    assert [(flag["curie"], flag["table"]) for flag in check["flags"]] == \
        [("icd10:Q99", None), ("icd10:T14.9", "B")]
    assert render_context(check, causes) == (
        "- Lower respiratory infections may be due to HIV (ACME Table E). "
        "Together they are coded as B20.8 (HIV disease resulting in other "
        "infectious and parasitic diseases).\n"
        "- Other chromosome abnormalities (Q99) is not in the ACME tables.\n"
        "- Injury (T14.9) can only be a contributing cause, not the "
        "underlying cause."
    )


def test_check_top_causes_both_ways(tmp_path):
    kg = make_kg(tmp_path)
    causes = {
        "icd10:J18.9": {"name": "Lower respiratory infections"},
        "icd10:E43": {"name": "Malnutrition"},
    }
    check = check_top_causes(kg, causes)
    # A pair linked both ways gives one link, so one line
    assert [(link["cause"], link["reverse"]["cause"])
            for link in check["links"]] == [("icd10:J18.9", "icd10:E43")]
    lines = render_context(check, causes).splitlines()
    assert len(lines) == len(check["links"]) + len(check["flags"])
    assert lines[0] == (
        "- Malnutrition may be due to Lower respiratory infections (ACME "
        "Table D, only in some cases), and Lower respiratory infections may "
        "be due to Malnutrition (ACME Table D). The order the case reports "
        "decides which is the underlying cause."
    )


def test_check_mccd_chain(tmp_path):
    kg = make_kg(tmp_path)
    check = check_mccd_chain(kg, ["J18.9", "B24", "", ""])
    assert [link["verdict"] for link in check["links"]] == ["accepted"]
    assert (check["underlying"], check["steps"]) == \
        ("B24", ["General Principle selects B24"])
    # Rule 2 selects pneumonia, which combines with HIV on a lower line
    check = check_mccd_chain(kg, ["J18.9", "C80.1", "B24"])
    assert [link["verdict"] for link in check["links"]] == \
        ["not_listed", "not_listed"]
    assert check["steps"] == ["Rule 2 selects J18.9",
                              "Table E DSC selects B20.8"]
    # Rules marked "maybe" are not applied but listed for review
    check = check_mccd_chain(kg, ["I50.9", "I10"])
    assert render_mccd_check(check) == (
        "- (a) I50.9 due to (b) I10: accepted only in some cases.\n"
        "Suggested underlying cause: I10 (General Principle selects I10). "
        "A suggestion, not official coding.\n"
        "Needs review: Table E LDC would select I11.0 only if: Must be "
        "hypertensive heart disease."
    )


def test_check_top_causes_catch_all_reverse(tmp_path, monkeypatch):
    kg = make_kg(tmp_path)
    monkeypatch.setattr(query, "CATCH_ALL_SIZE", 2)
    causes = {
        "icd10:J18.9": {"name": "Lower respiratory infections"},
        "icd10:D64.9": {"name": "Anemias"},
        "icd10:R99": {"name": "Undetermined"},
    }
    check = check_top_causes(kg, causes)
    # A catch-all rule allowing the reverse leaves the order to the case, and
    # an ill-defined cause links to nothing
    assert [(link["cause"], link["reverse"]["verdict"])
            for link in check["links"]] == \
        [("icd10:J18.9", "accepted_catch_all")]
    assert render_context(check, causes) == (
        "- Anemias may be due to Lower respiratory infections (ACME Table D), "
        "and Lower respiratory infections may be due to Anemias (ACME Table "
        "D, by a broad rule). The order the case reports decides which is "
        "the underlying cause."
    )
    # A "maybe" rule through a catch-all range isn't specific either
    causes = {"icd10:D64.9": {"name": "Anemias"},
              "icd10:E43": {"name": "Malnutrition"}}
    assert check_top_causes(kg, causes)["links"] == []


def test_contributing_cause_is_never_underlying(tmp_path):
    kg = make_kg(tmp_path)
    causes = {"icd10:E43": {"name": "Malnutrition"},
              "icd10:T14.9": {"name": "Injury"}}
    check = check_top_causes(kg, causes)
    assert "reverse" in check["links"][0]
    assert render_context(check, causes).splitlines()[0] == (
        "- Injury may be due to Malnutrition (ACME Table D). If both are "
        "reported, Malnutrition is the underlying cause."
    )
