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

def _slab_score(fragment: ProcessedFragment) -> float:
    """Compute a per-fragment 'slab-like' flatness score in [0, 1].

    A slab has one very small extent relative to the other two.
    Score = 1 - (min_extent / max_extent), so a perfect slab → 1.0,
    a cube → 0.0.

    Args:
        fragment: processed fragment with bounding box extent.

    Returns:
        Flatness score in [0, 1].
    """
    ext = np.sort(np.abs(fragment.extent.astype(float)))  # ascending
    if ext[-1] < 1e-6:
        return 0.0
    return float(1.0 - ext[0] / ext[-1])


def _build_graph(
    fragments: List[ProcessedFragment],
    pairwise_results: List[Dict[str, Any]],
    min_score: float,
    config: Optional[Dict[str, Any]] = None,
) -> "nx.Graph":
    """Build compatibility graph from pairwise registration results.

    Nodes are fragment names. Edges are added only when overlap_score exceeds
    min_score_threshold AND collision_fraction (if present in result) is below
    the reject threshold recorded in the result dict. Pairs already rejected by
    pairwise_match will not appear in pairwise_results, but we keep this as a
    secondary guard in case results were loaded from a file.

    A soft slab bonus is applied to edge weights: fragments that are already
    slab-like (thin relative to their longest dimension) receive a small upward
    adjustment to their edge weight, biasing the MST toward a stele-like assembly.

    Args:
        fragments: all processed fragments (to ensure all nodes exist).
        pairwise_results: output of compute_pairwise_matches.
        min_score: minimum overlap_score to include an edge.
        config: pipeline config dict (reads assembly.slab_bonus_weight).

    Returns:
        networkx Graph.
    """
    slab_bonus_weight = 0.0
    if config is not None:
        slab_bonus_weight = float(
            config.get("assembly", {}).get("slab_bonus_weight", 0.10)
        )

    G = nx.Graph()
    frag_map = {f.name: f for f in fragments}
    slab_scores: Dict[str, float] = {}
    for frag in fragments:
        G.add_node(frag.name, num_triangles=frag.num_triangles)
        slab_scores[frag.name] = _slab_score(frag)

    n_edges = 0
    n_skipped_score = 0
    n_skipped_collision = 0

    for r in pairwise_results:
        score = float(r["overlap_score"])
        if score <= min_score:
            n_skipped_score += 1
            continue

        # Secondary guard: also skip if collision_fraction is flagged high
        # (pairs already rejected by pairwise_match won't appear, but if
        # results come from a file without pre-filtering, this catches them)
        collision_frac = float(r.get("collision_fraction", 0.0))
        if collision_frac > 0.35:
            n_skipped_collision += 1
            logger.debug(
                "Graph: skipping edge %s/%s — collision_fraction=%.2f",
                r["fragment_a"][-8:], r["fragment_b"][-8:], collision_frac,
            )
            continue

        # Soft slab bonus: mean slab score of the two fragments multiplied by
        # the bonus weight — a small upward nudge, not a hard constraint.
        fa, fb = r["fragment_a"], r["fragment_b"]
        if slab_bonus_weight > 0.0:
            mean_slab = (slab_scores.get(fa, 0.0) + slab_scores.get(fb, 0.0)) / 2.0
            score = score * (1.0 + slab_bonus_weight * mean_slab)

        G.add_edge(
            fa, fb,
            weight=score,
            transform=np.array(r["transform"]),      # maps fragment_a -> fragment_b frame
            icp_fitness=float(r.get("icp_fitness", score)),
            icp_rmse=float(r.get("icp_rmse", 0.0)),
            collision_fraction=collision_frac,
            fragment_a=fa,
            fragment_b=fb,
        )
        n_edges += 1

    logger.info(
        "Compatibility graph: %d nodes  %d edges  "
        "(skipped: %d below score, %d high collision; threshold=%.2f; slab_bonus=%.2f)",
        G.number_of_nodes(), n_edges,
        n_skipped_score, n_skipped_collision, min_score, slab_bonus_weight,
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

    G = _build_graph(fragments, pairwise_results, min_score, config=config)
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

    # --- Diagnostics ---
    n_placed = sum(
        1 for p in placements.values()
        if p["status"] in ("placed", "anchor", "sub_anchor")
    )
    n_unplaced = len(fragments) - n_placed

    # Count MST components (each sub_anchor is a component root)
    n_components = sum(
        1 for p in placements.values()
        if p["status"] in ("anchor", "sub_anchor")
    )

    # Identify low-support placements: placed via only one MST edge with
    # no cross_validation support (cross_validation_fitness is None)
    low_support = [
        name for name, p in placements.items()
        if p["status"] == "placed"
        and p.get("cross_validation_fitness") is None
        and (p.get("pairwise_fitness") or 0.0) < 0.25
    ]

    logger.info(
        "Assembly complete: %d placed  %d unplaced  %d component(s)  %.1fs",
        n_placed, n_unplaced, n_components, time.perf_counter() - t0,
    )
    if low_support:
        logger.warning(
            "Low-confidence placements (single weak edge, no cross-validation): %s",
            ", ".join(n.split("_FR_")[-1][:5] for n in low_support),
        )
    if n_components > 1:
        logger.warning(
            "%d disconnected components — many fragments could not be linked to the anchor",
            n_components,
        )

    return {
        "method": "greedy_mst",
        "anchor": anchor,
        "placements": placements,
        "pairwise_results": pairwise_results,
        "diagnostics": {
            "n_placed": n_placed,
            "n_unplaced": n_unplaced,
            "n_components": n_components,
            "low_support_placements": low_support,
        },
    }
