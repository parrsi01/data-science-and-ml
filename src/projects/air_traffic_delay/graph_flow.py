"""Network graph analytics for route flow and congestion bottlenecks."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def build_route_graph(routes_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed airport route graph from routes."""

    graph = nx.DiGraph()
    for row in routes_df.itertuples(index=False):
        graph.add_edge(
            str(row.dep),
            str(row.arr),
            distance_km=float(row.distance_km),
            base_congestion=float(row.base_congestion),
            weight=1.0 + float(row.base_congestion),
        )
    return graph


def compute_graph_metrics(graph: nx.DiGraph) -> pd.DataFrame:
    """Compute operational graph metrics for each airport node."""

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "airport",
                "in_degree",
                "out_degree",
                "betweenness_centrality",
                "pagerank",
                "clustering",
            ]
        )

    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    pagerank = nx.pagerank(graph, alpha=0.85, weight="weight")
    clustering = nx.clustering(graph.to_undirected())

    rows = []
    for node in sorted(graph.nodes()):
        rows.append(
            {
                "airport": str(node),
                "in_degree": float(in_deg.get(node, 0)),
                "out_degree": float(out_deg.get(node, 0)),
                "betweenness_centrality": float(betweenness.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
                "clustering": float(clustering.get(node, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def attach_node_metrics_to_flights(
    flights_df: pd.DataFrame, node_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Attach departure and arrival node metrics onto flight rows."""

    dep_metrics = node_metrics.rename(
        columns={
            "airport": "dep",
            "in_degree": "dep_in_degree",
            "out_degree": "dep_out_degree",
            "betweenness_centrality": "dep_betweenness_centrality",
            "pagerank": "dep_pagerank",
            "clustering": "dep_clustering",
        }
    )
    arr_metrics = node_metrics.rename(
        columns={
            "airport": "arr",
            "in_degree": "arr_in_degree",
            "out_degree": "arr_out_degree",
            "betweenness_centrality": "arr_betweenness_centrality",
            "pagerank": "arr_pagerank",
            "clustering": "arr_clustering",
        }
    )
    merged = flights_df.merge(dep_metrics, on="dep", how="left").merge(arr_metrics, on="arr", how="left")
    return merged


def save_graph_artifacts(
    graph: nx.DiGraph,
    metrics_df: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Save graph metrics CSV and route graph visualization."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "graph_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=42, k=0.8)
    node_size = [6000 * max(metrics_df.set_index("airport").loc[node, "pagerank"], 0.005) for node in graph.nodes()]
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, arrows=True, arrowsize=10, width=1.0)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size, node_color="#0B6E4F", alpha=0.85)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7, font_color="white")
    ax.set_title("Route Network Graph (node size ~ PageRank)")
    ax.axis("off")
    fig.tight_layout()
    graph_png = output_dir / "route_graph.png"
    fig.savefig(graph_png, dpi=150)
    plt.close(fig)

    return {"graph_metrics_csv": str(metrics_path), "route_graph_png": str(graph_png)}

