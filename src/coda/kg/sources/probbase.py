"""
This module gets the list of VA questions from the "probbase" associated
with InterVA and construct nodes for the questions themselves, and
build edges between the questions and the causes they relate to
using the information in the probbase. It also exports the probbase's skip
rules as edges between questions.
"""

__all__ = ["ProbBaseExporter"]

import re
import pandas as pd

from coda.kg.sources import KGSourceExporter, write_tsv_gz

PROBBASE_URL = (
    "https://github.com/verbal-autopsy-software/interva/raw/"
    "refs/heads/main/src/interva/data/probbase.xls"
)

RULE_CODE = re.compile(r"^(i\d{3}[a-z])([YN])$")
RULE_COLUMNS = {
    "ask_if": ["doaskif", "nnonly"],
    "dont_ask_if": [f"dontask{i}" for i in range(1, 9)],
}


def process_va_col(col_name):
    code = col_name[2:]
    if code.endswith("00"):
        code = code[:-2]
    else:
        code = f"{code[:2]}.{code[2:]}"
    return f"who.va:VAs-{code}"


def rule_edges(df):
    """Edges for the probbase skip rules, as [start, end, type, answer] rows."""
    questions = df.dropna(subset=["indic"])
    curie_by_indic = dict(zip(questions["indic"], questions["who_curie"]))
    edges = []
    for _, row in questions.iterrows():
        for edge_type, columns in RULE_COLUMNS.items():
            for col in columns:
                if pd.isna(row[col]) or not str(row[col]).strip():
                    continue
                match = RULE_CODE.match(str(row[col]).strip())
                if not match:
                    raise ValueError(f"Unrecognized rule code {row[col]!r} in {col}")
                indic, answer = match.groups()
                edges.append([row["who_curie"], curie_by_indic[indic], edge_type, answer])
    return edges


class ProbBaseExporter(KGSourceExporter):
    name = "probbase"

    def export(self):
        df = pd.read_excel(PROBBASE_URL, sheet_name="probbase")
        id_column = "who_2016"
        name_column = "qdesc"
        prop_columns = ["indic", "sdesc", "ilab", "subst", "samb"]
        df["who_curie"] = df[id_column].apply(lambda x: f"who.va.q:{x}")
        va_question_cols = {
            col: process_va_col(col) for col in df.columns if col.startswith("b_")
        }
        nodes = df[["who_curie", name_column, *prop_columns]].copy()
        nodes[":LABEL"] = "who.va.q"
        nodes = nodes.rename(columns={"who_curie": "id:ID", name_column: "name"})
        nodes = nodes.dropna(subset=["indic"])

        write_tsv_gz(
            nodes.sort_values("id:ID").drop_duplicates(), self.nodes_file
        )

        edges = []
        for _, row in df.iterrows():
            if pd.isna(row["indic"]):
                continue
            node_curie = row["who_curie"]
            for col, va_curie in va_question_cols.items():
                edge = [
                    node_curie,
                    va_curie,
                    "probbase_rel",
                    row[col],
                ]
                edges.append(edge)
        edges.extend(rule_edges(df))
        edge_df = pd.DataFrame(
            edges, columns=[":START_ID", ":END_ID", ":TYPE", "value"]
        )
        edge_df = pd.concat(
            [edge_df.drop(columns=[":TYPE"]), edge_df[":TYPE"].apply(pd.Series).rename(columns={0: ':TYPE'}) ],
            axis=1,
        )
        edge_df = edge_df.sort_values([":START_ID", ":END_ID", ":TYPE", "value"])
        write_tsv_gz(edge_df.drop_duplicates(), self.edges_file)


if __name__ == "__main__":
    exporter = ProbBaseExporter()
    exporter.export()
