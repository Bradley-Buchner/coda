import pandas as pd

from coda.kg import query
from coda.kg.check import check_top_causes, render_context
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
    ["icd10:J18.9", "acme:table_A", "in_code_list", "", "", "", ""],
    ["icd10:T14.9", "acme:table_B", "in_code_list", "", "", "", ""],
]


def make_kg(tmp_path):
    edges = pd.DataFrame(EDGES, columns=[
        ":START_ID", ":END_ID", ":TYPE", "maybe:boolean", "rule_type",
        "result", "reason",
    ])
    write_tsv_gz(edges, tmp_path / "acme_edges.tsv.gz")
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
        "Together they are coded as B20.8.\n"
        "- Other chromosome abnormalities (Q99) is not in the ACME tables.\n"
        "- Injury (T14.9) can only be a contributing cause, not the "
        "underlying cause."
    )
