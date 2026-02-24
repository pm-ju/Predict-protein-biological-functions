"""
Stage 3 — Graph Construction
Build a Sequence Similarity Network (SSN) from the edge list,
compute graph-theoretic node features, and run community detection.
"""

import csv
import json
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import log


def build_ssn(
    edges_csv: Path = None,
    proteins_csv: Path = None,
    identity_threshold: float = None,
    output_dir: Path = None,
) -> tuple[nx.Graph, Path]:
    """
    Build a Sequence Similarity Network from an edge list.
    
    Returns (graph, graph_features_csv_path).
    """
    identity_threshold = identity_threshold or config.EDGE_WEIGHT_THRESHOLD
    output_dir = output_dir or config.GRAPH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_csv = edges_csv or config.SIMILARITY_DIR / f"edges_t{int(identity_threshold*100)}.csv"
    proteins_csv = proteins_csv or config.DATA_DIR / "proteins.csv"
    graph_features_csv = output_dir / "graph_features.csv"
    graph_file = output_dir / "ssn.graphml"
    stats_json = output_dir / "graph_stats.json"

    # ── Load all protein accessions (nodes) ───────────────────────────────
    all_nodes = []
    with open(proteins_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_nodes.append(row["accession"])

    log.info(f"Loaded {len(all_nodes)} protein accessions.")

    # ── Build graph ───────────────────────────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from(all_nodes)

    edge_count = 0
    with open(edges_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            identity = float(row["identity"])
            if identity >= identity_threshold:
                G.add_edge(
                    row["source"], row["target"],
                    weight=identity,
                    evalue=float(row.get("evalue", 0)),
                )
                edge_count += 1

    log.info(f"Built SSN: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # ── Compute graph-theoretic features ──────────────────────────────────
    log.info("Computing graph-theoretic node features…")

    # Degree
    degrees = dict(G.degree())

    # Clustering coefficient
    clustering = nx.clustering(G)

    # Connected components
    components = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            components[node] = i
    num_components = i + 1 if G.number_of_nodes() > 0 else 0

    # Betweenness centrality (can be slow for large graphs)
    if config.COMPUTE_CENTRALITY and G.number_of_nodes() <= 5000:
        log.info("Computing betweenness centrality…")
        betweenness = nx.betweenness_centrality(G)
    else:
        log.info("Skipping betweenness centrality (graph too large or disabled).")
        betweenness = {n: 0.0 for n in G.nodes()}

    # Community detection (Louvain)
    communities = {}
    if config.RUN_COMMUNITY_DETECTION:
        try:
            import community as community_louvain
            log.info("Running Louvain community detection…")
            partition = community_louvain.best_partition(G, random_state=config.RANDOM_SEED)
            communities = partition
            num_communities = len(set(partition.values()))
            log.info(f"Found {num_communities} communities.")
        except ImportError:
            log.warning("python-louvain not installed. Skipping community detection.")
            communities = {n: 0 for n in G.nodes()}
    else:
        communities = {n: 0 for n in G.nodes()}

    # ── Save graph features ───────────────────────────────────────────────
    rows = []
    for node in all_nodes:
        rows.append({
            "accession": node,
            "degree": degrees.get(node, 0),
            "clustering_coeff": round(clustering.get(node, 0.0), 6),
            "betweenness": round(betweenness.get(node, 0.0), 6),
            "component_id": components.get(node, -1),
            "community_id": communities.get(node, -1),
        })

    with open(graph_features_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "degree", "clustering_coeff",
                                                "betweenness", "component_id", "community_id"])
        writer.writeheader()
        writer.writerows(rows)

    # ── Save graph ────────────────────────────────────────────────────────
    nx.write_graphml(G, str(graph_file))
    log.info(f"Saved graph to {graph_file}")

    # ── Statistics ────────────────────────────────────────────────────────
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "num_connected_components": num_components,
        "avg_degree": round(2 * G.number_of_edges() / max(G.number_of_nodes(), 1), 2),
        "avg_clustering": round(nx.average_clustering(G), 6),
        "num_communities": len(set(communities.values())),
    }

    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Graph statistics: {json.dumps(stats, indent=2)}")
    return G, graph_features_csv


def load_graph(output_dir: Path = None) -> nx.Graph:
    """Load a previously saved SSN."""
    output_dir = output_dir or config.GRAPH_DIR
    graph_file = output_dir / "ssn.graphml"
    return nx.read_graphml(str(graph_file))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Build Sequence Similarity Network")
    parser.add_argument("--threshold", type=float, default=config.EDGE_WEIGHT_THRESHOLD)
    args = parser.parse_args()
    build_ssn(identity_threshold=args.threshold)
