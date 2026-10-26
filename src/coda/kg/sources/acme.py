import pandas as pd

from openacme.acme import get_acme_tables
from openacme.icd10 import Icd10Graph
from coda.kg.sources import KGSourceExporter, write_tsv_gz


class ACMEExporter(KGSourceExporter):
    name = "acme"

    def export(self):
        tables = get_acme_tables()
        g = tables["graph"]
        # Add the ranges of Table E that are not in Table D
        icd10_graph = Icd10Graph()
        for rule in tables["table_e"]:
            source = rule["source"]
            if isinstance(source, tuple) and source not in g:
                g.add_node(source, kind="range")
                for code in icd10_graph.expand_icd10_range(source[0], source[1]):
                    if code not in g:
                        g.add_node(code, kind="category")
                    g.add_edge(code, source, kind="part_of_range")
        # We need to make sure all nodes have an `icd10:` prefix
        # in their label
        nodes = []
        edges = []
        for node, data in g.nodes(data=True):
            nodes.append(
                [
                    f"icd10:{node}",  # id:ID
                    "icd10",  # :LABEL
                    data.get("kind"),  # class_kind
                    node,  # code
                ]
            )
        # Nodes for the code lists of Tables A, B, C1-C3 and H
        for name in tables["code_lists"]:
            nodes.append([f"acme:table_{name}", "acme", "code_list", name])
        nodes_df = pd.DataFrame(
            nodes,
            columns=["id:ID", ":LABEL", "class_kind", "code"],
        )
        write_tsv_gz(nodes_df.sort_values("id:ID"), self.nodes_file)

        for source, target, data in g.edges(data=True):
            edges.append(
                [
                    f"icd10:{source}",  # :START_ID
                    f"icd10:{target}",  # :END_ID
                    data.get("kind", "related_to"),  # :TYPE
                    data.get("maybe"),  # maybe:boolean
                ]
            )
        edges_df = pd.DataFrame(
            edges, columns=[":START_ID", ":END_ID", ":TYPE", "maybe:boolean"]
        )

        # Table E modification rules, where the target is the tentative
        # underlying cause and the source is another condition
        rules = []
        for rule in tables["table_e"]:
            rules.append(
                [
                    f"icd10:{rule['source']}",  # :START_ID
                    f"icd10:{rule['target']}",  # :END_ID
                    "acme_modification",  # :TYPE
                    rule["maybe"],  # maybe:boolean
                    rule["rule_type"],  # rule_type
                    rule["result"],  # result
                    rule["hash"],  # hash:boolean
                    rule["reason"],  # reason
                ]
            )
        rules_df = pd.DataFrame(
            rules,
            columns=[":START_ID", ":END_ID", ":TYPE", "maybe:boolean",
                     "rule_type", "result", "hash:boolean", "reason"],
        )

        code_list_edges = []
        for name, codes in tables["code_lists"].items():
            for code in codes:
                code_list_edges.append(
                    [f"icd10:{code}", f"acme:table_{name}", "in_code_list"]
                )
        code_list_df = pd.DataFrame(
            code_list_edges, columns=[":START_ID", ":END_ID", ":TYPE"]
        )

        edges_df = pd.concat([edges_df, rules_df, code_list_df])
        write_tsv_gz(
            edges_df.drop_duplicates().sort_values([":START_ID", ":END_ID"]),
            self.edges_file,
        )


if __name__ == "__main__":
    exporter = ACMEExporter()
    exporter.export()
