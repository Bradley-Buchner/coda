import pandas as pd
import pytest

from coda.kg.graph import KGView
from coda.kg.sources import write_tsv_gz


def write(path, rows, columns):
    write_tsv_gz(pd.DataFrame(rows, columns=columns), path)


@pytest.fixture
def graph(tmp_path):
    write(tmp_path / "icd10_nodes.tsv.gz", [
        ("icd10:A00", "Cholera"),
        ("icd10:A00.0", "Cholera due to Vibrio cholerae 01"),
        ("icd10:A00.9", "Cholera, unspecified"),
        ("icd10:J18.9", "Pneumonia, unspecified"),
        ("icd10:K35.2", "Acute appendicitis"),
    ], ["id:ID", "name"])
    write(tmp_path / "icd10_edges.tsv.gz", [
        ("icd10:A00.0", "icd10:A00", "is_a"),
        ("icd10:A00.9", "icd10:A00", "is_a"),
        ("icd10:J18.9", "icd10:J18", "is_a"),
    ], [":START_ID", ":END_ID", ":TYPE"])
    narrow, wide = "icd10:('A00.0', 'A00.9')", "icd10:('A00.0', 'J18.9')"
    write(tmp_path / "acme_edges.tsv.gz", [
        ("icd10:A00.0", narrow, "part_of_range"),
        ("icd10:A00.9", narrow, "part_of_range"),
        ("icd10:A00.0", wide, "part_of_range"),
        (narrow, "icd10:K35.2", "causes"),
        (narrow, "icd10:A00.9", "causes"),
        (wide, "icd10:J18.9", "causes"),
        ("icd10:J18.9", "icd10:K35.2", "causes"),
        ("icd10:R82.6", "icd10:J18.9", "causes"),
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
        ("who.va.q:diarrhea", "Did (s)he have diarrhea?"),
        ("who.va.q:cough", "Did (s)he have a cough?"),
        ("who.va.q:female", "Was she female?"),
    ], ["id:ID", "name"])
    write(tmp_path / "probbase_edges.tsv.gz", [
        ("who.va.q:diarrhea", "who.va:VAs-01.04", "A+", "probbase_rel"),
        ("who.va.q:cough", "who.va:VAs-01.04", "C", "probbase_rel"),
        ("who.va.q:female", "who.va:VAs-01.04", "A", "probbase_rel"),
        ("who.va.q:female", "who.va:VAs-01.02", "A", "probbase_rel"),
        ("who.va.q:cough", "who.va:VAs-01.02", "", "dont_ask_if"),
    ], [":START_ID", ":END_ID", "value", ":TYPE"])
    return KGView(tmp_path)


def test_acme_keeps_only_narrow_ranges(graph):
    # The narrow range is expanded to its codes, without a self-loop; the
    # wide range and the R-chapter edge are dropped
    assert graph.neighbors("icd10:K35.2", "causes", "in") == \
        ["icd10:A00.0", "icd10:A00.9", "icd10:J18.9"]
    assert graph.neighbors("icd10:A00.9", "causes", "in") == ["icd10:A00.0"]
    assert graph.neighbors("icd10:J18.9", "causes", "in") == []


def test_relations_exclude_maps_to(graph):
    assert graph.relations("icd10:J18.9") == \
        [("is_a", "out"), ("causes", "out")]


def test_who_va_causes_walks_up_hierarchy(graph):
    assert graph.who_va_causes("A00.0") == ["who.va:VAs-01.04"]
    assert graph.who_va_causes("K35.2") == []


def test_strong_findings_drop_weak_and_uninformative(graph):
    assert graph.strong_findings("who.va:VAs-01.04") == \
        [("who.va.q:diarrhea", "A+")]
