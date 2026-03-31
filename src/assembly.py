"""Global fragment assembly: MST-based transform chaining (Phase 5) and
post-placement cross-validation (Phase 6)."""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

from src.preprocess import ProcessedFragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph(
    fragments: List[ProcessedFragment],
    pairwise_results: List[Dict[str, Any]],
    min_score: float,
) -> "nx.Graph":
    """Build compatibility graph from pairwise registration results.

    Nodes are fragment names. Edges are added only when overlap_score exceeds
    min_score_threshold. Edge weight = overlap_score.

    Args:
        fragments: all processed fragments (to ensure all nodes exist).
        pairwise_results: output of compute_pairwise_matches.
        min_score: minimum overlap_score to include an edge.

    Returns:
        networkx Graph.
    """
    G = nx.Graph()
    for frag in fragments:
        G.add_node(frag.name, num_triangles=frag.num_triangles)

    n_edges = 0
    for r in pairwise_results:
        score = float(r["overlap_score"])
        if score > min_score:
            G.add_edge(
                r["fragment_a"], r["fragment_b"],
                weight=score,
                transform=np.array(r["transform"]),      # maps fragment_a -> fragment_b frame
                icp_fitness=float(r.get("icp_fitness", score)),
                icp_rmse=float(r.get("icp_rmse", 0.0)),
                fragment_a=r["fragment_a"],
                fragment_b=r["fragment_b"],
            )
            n_edges += 1

    logger.info(
        "Compatibility graph: %d nodes  %d edges (threshold=%.2f)",
        G.number_of_nodes(), n_edges, min_score,
    )
    return G


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

def _select_anchor(fragments: List[ProcessedFragment], config: Dict[str, Any]) -> str:
    """Select the anchor fragment by policy.

    Policies:
      largest — fragment with most triangles (most distinctive geometry).
      index   — fragment at assembly.anchor_index position in the list.

    Args:
        fragments: all processed fragments.
        config: pipeline config dict.

    Returns:
        Name of the chosen anchor fragment.
    """
    policy = config["assembly"].get("anchor_by", "largest")
    if policy == "index":
        idx = int(config["assembly"].get("anchor_index", 0))
        return fragments[min(idx, len(fragments) - 1)].name
    return max(fragments, key=lambda f: f.num_triangles).name


# ---------------------------------------------------------------------------
# Transform chain composition (BFS over MST)
# ---------------------------------------------------------------------------

def _compose_transforms(
    G: "nx.Graph",
    mst: "nx.Graph",
    anchor: str,
    fragments: List[ProcessedFragment],
) -> Dict[str, Dict[str, Any]]:
    """BFS from each component's anchor to build global transforms.

    Transform convention (per design_v2):
        T maps source (fragment_a) points to target (fragment_b) frame.
        For edge (parent, child):
          - child == fragment_a  =>  T maps child -> parent
                                      child_global = T_parent @ T
          - child == fragment_b  =>  T maps parent -> child
                                      child_global = T_parent @ inv(T)

    Each disconnected component in the MST gets its own sub-anchor (largest
    fragment in that component). The global anchor always gets identity.

    Args:
        G: full compatibility graph (holds edge metadata).
        mst: maximum spanning tree subgraph.
        anchor: name of the primary anchor fragment.
        fragments: processed fragments (for num_triangles lookup).

    Returns:
        Dict mapping fragment name -> placement dict.
    """
    frag_map = {f.name: f for f in fragments}
    placements: Dict[str, Dict[str, Any]] = {}

    components = list(nx.connected_components(mst))
    logger.info("MST: %d nodes  %d edges  %d component(s)",
                mst.number_of_nodes(), mst.number_of_edges(), len(components))

    for comp_idx, component in enumerate(components):
        # Choose the component anchor
        if anchor in component:
            comp_anchor = anchor
            anchor_status = "anchor"
        else:
            comp_anchor = max(
                (name for name in component if name in frag_map),
                key=lambda name: frag_map[name].num_triangles,
                default=next(iter(component)),
            )
            anchor_status = "sub_anchor"
            logger.info("Component %d: sub-anchor = %s", comp_idx + 1, comp_anchor)

        placements[comp_anchor] = {
            "transform": np.eye(4),
            "status": anchor_status,
            "placed_via": None,
            "pairwise_fitness": None,
            "pairwise_rmse": None,
            "cross_validation_fitness": None,
        }

        # BFS
        queue = deque([comp_anchor])
        visited = {comp_anchor}

        while queue:
            parent = queue.popleft()
            T_parent = placements[parent]["transform"]

            for neighbour in mst.neighbors(parent):
                if neighbour in visited:
                    continue
                visited.add(neighbour)

                edge = G[parent][neighbour]
                T_edge = edge["transform"]   # maps fragment_a -> fragment_b
                fa = edge["fragment_a"]

                if neighbour == fa:
                    # T_edge: neighbour -> parent
                    T_child_global = T_parent @ T_edge
                else:
                    # T_edge: parent -> neighbour  =>  inv maps neighbour -> parent
                    T_child_global = T_parent @ np.linalg.inv(T_edge)

                placements[neighbour] = {
                    "transform": T_child_global,
                    "status": "placed",
                    "placed_via": parent,
                    "pairwise_fitness": float(edge["icp_fitness"]),
                    "pairwise_rmse": float(edge["icp_rmse"]),
                    "cross_validation_fitness": None,
                }
                queue.append(neighbour)

    return placements


# ---------------------------------------------------------------------------
# Phase 6 — Post-placement cross-validation
# ---------------------------------------------------------------------------

def _cross_validate(
    placements: Dict[str, Dict[str, Any]],
    frag_map: Dict[str, ProcessedFragment],
    config: Dict[str, Any],
) -> None:
    """Validate each placed fragment against nearby already-placed fragments.

    For each placed fragment F, find all placed fragments N within
    proximity_check_radius (in assembly space). Compute ICP fitness between
    F_transformed and N_transformed. Record mean supporting fitness in
    placements[name]["cross_validation_fitness"].

    Results are DIAGNOSTIC only — they do NOT override the MST placement.

    Args:
        placements: placement dict from _compose_transforms (modified in-place).
        frag_map: fragment name -> ProcessedFragment.
        config: pipeline config dict.
    """
    asm_cfg = config["assembly"]
    proximity_r = float(asm_cfg.get("proximity_check_radius", 50.0))
    min_fit = float(asm_cfg.get("cross_validation_min_fitness", 0.1))
    icp_corr = float(config["icp"]["max_correspondence_distance"])

    placed_names = [
        n for n, p in placements.items()
        if p.get("transform") is not None
    ]

    # Pre-compute assembly-frame centroids for proximity check
    centroids: Dict[str, np.ndarray] = {}
    for name in placed_names:
        T = placements[name]["transform"]
        c = np.append(frag_map[name].center, 1.0)
        centroids[name] = (T @ c)[:3]

    for name in placed_names:
        if placements[name]["status"] in ("anchor", "sub_anchor"):
            continue

        T_f = placements[name]["transform"]
        # Build transformed source PCD (points only; normals not needed for evaluate_registration)
        src_pts = (T_f[:3, :3] @ np.asarray(frag_map[name].pcd.points).T + T_f[:3, 3:4]).T
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_pts)

        support_fitnesses: List[float] = []
        for other in placed_names:
            if other == name:
                continue
            if np.linalg.norm(centroids[name] - centroids[other]) > proximity_r:
                continue

            T_o = placements[other]["transform"]
            tgt_pts = (T_o[:3, :3] @ np.asarray(frag_map[other].pcd.points).T + T_o[:3, 3:4]).T
            tgt_pcd = o3d.geometry.PointCloud()
            tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)

            reg = o3d.pipelines.registration.evaluate_registration(
                src_pcd, tgt_pcd, icp_corr, np.eye(4),
            )
            if reg.fitness >= min_fit:
                support_fitnesses.append(float(reg.fitness))

        if support_fitnesses:
            placements[name]["cross_validation_fitness"] = float(np.mean(support_fitnesses))


# ---------------------------------------------------------------------------
# Phase 5+6 orchestration
# ---------------------------------------------------------------------------

def assemble_fragments(
    fragments: List[ProcessedFragment],
    pairwise_results: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build MST-based global assembly from pairwise registration results.

    Phase 5:
      1. Build compatibility graph (edges with overlap_score > threshold).
      2. Compute maximum spanning tree.
      3. BFS from anchor to compose 4x4 global transforms.

    Phase 6 (if validate_against_placed is true):
      4. Cross-validate each placed fragment against nearby placed neighbours.
         Stores mean supporting fitness for diagnostic purposes only.

    Fragments not connected to any neighbour above the threshold are marked
    "unplaced" — the pipeline does not fail on disconnected fragments.

    Args:
        fragments: preprocessed fragments (Phase 0+1 output).
        pairwise_results: output of compute_pairwise_matches.
        config: pipeline config dict.

    Returns:
        Dict with keys: method, anchor, placements, pairwise_results.
        placements maps fragment_name -> dict with transform (4x4 ndarray),
        status, placed_via, pairwise_fitness, pairwise_rmse,
        cross_validation_fitness.
    """
    if not _HAS_NX:
        raise RuntimeError(
            "networkx is required for assembly. Install with: pip install networkx"
        )

    t0 = time.perf_counter()
    asm_cfg = config["assembly"]
    min_score = float(asm_cfg["min_score_threshold"])

    G = _build_graph(fragments, pairwise_results, min_score)
    mst = nx.maximum_spanning_tree(G, weight="weight")
    anchor = _select_anchor(fragments, config)
    logger.info("Anchor: %s", anchor)

    placements = _compose_transforms(G, mst, anchor, fragments)

    # Mark unplaced fragments (isolated nodes — no edge above threshold)
    for frag in fragments:
        if frag.name not in placements:
            placements[frag.name] = {
                "transform": None,
                "status": "unplaced",
                "reason": "no_match_above_threshold",
                "placed_via": None,
                "pairwise_fitness": None,
                "pairwise_rmse": None,
                "cross_validation_fitness": None,
            }

    # Phase 6: cross-validation
    if bool(asm_cfg.get("validate_against_placed", True)):
        frag_map = {f.name: f for f in fragments}
        _cross_validate(placements, frag_map, config)

    n_placed = sum(
        1 for p in placements.values()
        if p["status"] in ("placed", "anchor", "sub_anchor")
    )
    logger.info(
        "Assembly complete: %d / %d placed  %.1fs",
        n_placed, len(fragments), time.perf_counter() - t0,
    )

    return {
        "method": "greedy_mst",
        "anchor": anchor,
        "placements": placements,
        "pairwise_results": pairwise_results,
    }
