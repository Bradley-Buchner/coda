import pandas as pd
import pytest

from coda.inference.tog.backward import walk_back
from coda.kg.graph import HUB_TARGETS, KGView
from coda.kg.sources import write_tsv_gz


def write(path, rows, columns):
    write_tsv_gz(pd.DataFrame(rows, columns=columns), path)


@pytest.fixture
def graph(tmp_path):
    write(tmp_path / "icd10_nodes.tsv.gz", [
        ("icd10:A00", "Cholera"),
        ("icd10:A00.0", "Cholera due to Vibrio cholerae 01"),
        ("icd10:A00.9", "Cholera, unspecified"),
        ("icd10:A00-A09", "Intestinal infectious diseases"),
        ("icd10:A09", "Gastroenteritis"),
        ("icd10:B24", "HIV disease"),
        ("icd10:J09-J18", "Influenza and pneumonia"),
        ("icd10:J18.9", "Pneumonia, unspecified"),
        ("icd10:K35.2", "Acute appendicitis"),
    ], ["id:ID", "name"])
    write(tmp_path / "icd10_edges.tsv.gz", [
        ("icd10:A00.0", "icd10:A00", "is_a"),
        ("icd10:A00.9", "icd10:A00", "is_a"),
        ("icd10:A00", "icd10:A00-A09", "is_a"),
        ("icd10:A09", "icd10:A00-A09", "is_a"),
        ("icd10:A00-A09", "icd10:I", "is_a"),
        ("icd10:J18.9", "icd10:J18", "is_a"),
        ("icd10:J18", "icd10:J09-J18", "is_a"),
    ], [":START_ID", ":END_ID", ":TYPE"])
    narrow, block = "icd10:('A00.0', 'A00.9')", "icd10:('A00.0', 'A09')"
    wide = "icd10:('A00.0', 'J18.9')"
    hub_edges = [("icd10:B24", f"icd10:K{i}", "causes")
                 for i in range(HUB_TARGETS)]
    write(tmp_path / "acme_edges.tsv.gz", [
        ("icd10:A00.0", narrow, "part_of_range"),
        ("icd10:A00.9", narrow, "part_of_range"),
        ("icd10:A00.0", block, "part_of_range"),
        ("icd10:A09", block, "part_of_range"),
        ("icd10:A00.0", wide, "part_of_range"),
        (narrow, "icd10:K35.2", "causes"),
        (narrow, "icd10:A00.9", "causes"),
        (block, "icd10:K35.2", "causes"),
        (wide, "icd10:J18.9", "causes"),
        ("icd10:J18.9", "icd10:K35.2", "causes"),
        ("icd10:R11", "icd10:J18.9", "causes"),
        ("icd10:R82.6", "icd10:J18.9", "causes"),
        ("icd10:R54", "icd10:J18.9", "causes"),
        ("icd10:B24", "icd10:J18.9", "causes"),
        *hub_edges,
    ], [":START_ID", ":END_ID", ":TYPE"])
    write(tmp_path / "icd11_nodes.tsv.gz", [
        ("icd11:ca40", "CA40", "Pneumonia"),
        ("icd11:1a40", "1A40", "Gastroenteritis"),
        ("icd11:1a40.0", "1A40.0", "Viral gastroenteritis"),
        ("icd11:da10", "DA10", "Appendicitis"),
        ("icd11:da11", "DA11", "Appendiceal abscess"),
    ], ["id:ID", "code", "name"])
    write(tmp_path / "icd11_edges.tsv.gz", [
        ("icd11:1a40.0", "icd11:1a40", "is_a"),
        ("icd11:ca40", "icd10:J09-J18", "maps_to"),
        ("icd11:1a40", "icd10:A09", "maps_to"),
        ("icd11:da10", "icd10:K35.2", "maps_to"),
        ("icd11:da11", "icd10:K35.2", "maps_to"),
    ], [":START_ID", ":END_ID", ":TYPE"])
    write(tmp_path / "who_va_nodes.tsv.gz", [
        ("who.va:VAs-01.04", "Diarrheal diseases"),
        ("who.va:VAs-01.02", "Acute respiratory infection"),
    ], ["id:ID", "name"])
    write(tmp_path / "who_va_edges.tsv.gz", [
        ("icd10:A00", "who.va:VAs-01.04", "maps_to"),
        ("icd10:J18.9", "who.va:VAs-01.02", "maps_to"),
    ], [":START_ID", ":END_ID", ":TYPE"])
    write(tmp_path / "probbase_nodes.tsv.gz", [
        ("who.va.q:diarrhea", "Did (s)he have diarrhea?", "Y"),
        ("who.va.q:cough", "Did (s)he have a cough?", "Y"),
        ("who.va.q:female", "Was she female?", "Y"),
    ], ["id:ID", "name", "subst"])
    write(tmp_path / "probbase_edges.tsv.gz", [
        ("who.va.q:diarrhea", "who.va:VAs-01.04", "A+", "probbase_rel"),
        ("who.va.q:cough", "who.va:VAs-01.04", "C", "probbase_rel"),
        ("who.va.q:female", "who.va:VAs-01.04", "A", "probbase_rel"),
        ("who.va.q:female", "who.va:VAs-01.02", "A", "probbase_rel"),
        ("who.va.q:cough", "who.va:VAs-01.02", "", "dont_ask_if"),
    ], [":START_ID", ":END_ID", "value", ":TYPE"])
    return KGView(tmp_path)


def test_acme_keeps_sources_within_one_block(graph):
    # Ranges stay single nodes, without the wide range, the R-chapter findings
    # and senility; the symptom R11 and the hub B24 stay
    assert graph.neighbors("icd10:K35.2", "causes", "in") == \
        ["icd10:('A00.0', 'A00.9')", "icd10:('A00.0', 'A09')", "icd10:J18.9"]
    assert graph.neighbors("icd10:J18.9", "causes", "in") == \
        ["icd10:R11", "icd10:B24"]
    assert graph.name("icd10:('A00.0', 'A09')") == \
        "A00.0–A09 Intestinal infectious diseases"
    # A range's edge to a code inside it says nothing
    assert graph.neighbors("icd10:A00.9", "causes", "in") == []


def test_code_causes_include_its_ranges_but_not_hubs(graph):
    assert graph.neighbors("icd10:A09", "causes", "out") == ["icd10:K35.2"]
    assert graph.neighbors("icd10:B24", "causes", "out") == []
    assert ("causes", "out") not in graph.relations("icd10:B24")


def test_icd11_names_only_one_to_one_matches(graph):
    assert graph.same_as("A09") == "ICD-11 1A40 Gastroenteritis"
    # Two ICD-11 codes map to K35.2, and J18.9's map stops at its block
    assert graph.same_as("K35.2") == ""
    assert graph.same_as("J18.9") == ""


@pytest.mark.asyncio
async def test_walk_names_the_match_but_stays_in_icd10(graph):
    calls = []

    async def call(step, system_prompt, user_prompt, schema, **context):
        calls.append({"user_prompt": user_prompt, **context})
        return {"chosen": [], "reason": ""}

    await walk_back(graph, call, "Case.", "Diarrheal Diseases", "A09")
    assert "same as: ICD-11 1A40 Gastroenteritis" in calls[0]["user_prompt"]
    assert not any("ICD-11" in fact for fact in calls[0]["options"])


def test_relations_exclude_maps(graph):
    assert graph.relations("icd10:J18.9") == [
        ("is_a", "out"), ("causes", "out"), ("causes", "in")]


def test_who_va_causes_walks_up_hierarchy(graph):
    assert graph.who_va_causes("A00.0") == ["who.va:VAs-01.04"]
    assert graph.who_va_causes("K35.2") == []


def test_strong_findings_drop_weak_and_uninformative(graph):
    assert graph.strong_findings("who.va:VAs-01.04") == \
        [("who.va.q:diarrhea", "A+")]
