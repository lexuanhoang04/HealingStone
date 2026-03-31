"""Metric computation and summary export (Phase 8)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

from src.preprocess import ProcessedFragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collision detection helper
# ---------------------------------------------------------------------------

def _collision_fraction(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    collision_dist: float,
) -> float:
    """Fraction of src points within collision_dist of any tgt point.

    Uses evaluate_registration with an identity transform: the two PCDs are
    already in the same (assembly) coordinate frame.

    Args:
        src_pts: (N, 3) array of source points in assembly frame.
        tgt_pts: (M, 3) array of target points in assembly frame.
        collision_dist: proximity threshold in mm.

    Returns:
        Fraction in [0, 1].
    """
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        return 0.0

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)

    reg = o3d.pipelines.registration.evaluate_registration(
        src_pcd, tgt_pcd, collision_dist, np.eye(4),
    )
    return float(reg.fitness)


# ---------------------------------------------------------------------------
# Global metrics
# ---------------------------------------------------------------------------

def compute_global_metrics(
    fragments: List[ProcessedFragment],
    assembly: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute global assembly metrics per design_v2 §C.

    Metrics computed:
      fragments_placed, placement_rate
      mean_icp_fitness, mean_icp_rmse  (over spanning-tree edges)
      assembly_bounding_box_mm         (XYZ extent of merged assembly cloud)
      max_collision_fraction           (worst-case over adjacent placed pairs)
      collision_flags                  (pairs flagged collision_fraction > 0.05)

    All metrics measure SELF-CONSISTENCY, not absolute correctness. A
    mean_icp_fitness of 0.30 is a reasonable result for real heritage data.

    Args:
        fragments: all processed fragments.
        assembly: output of assemble_fragments (or refine_assembly["assembly"]).
        config: pipeline config dict.

    Returns:
        Dict of global metric values.
    """
    placements = assembly.get("placements", {})
    pairwise = assembly.get("pairwise_results", [])
    frag_map = {f.name: f for f in fragments}
    collision_dist = float(config["assembly"].get("collision_distance_mm", 2.0))

    n_total = len(fragments)
    placed_names = [
        n for n, p in placements.items()
        if p.get("transform") is not None
        and p.get("status") in ("placed", "anchor", "sub_anchor")
    ]
    n_placed = len(placed_names)
    placement_rate = n_placed / n_total if n_total > 0 else 0.0

    # Mean fitness / RMSE over spanning-tree edges
    fitnesses = [
        p["pairwise_fitness"]
        for p in placements.values()
        if p.get("pairwise_fitness") is not None
    ]
    rmses = [
        p["pairwise_rmse"]
        for p in placements.values()
        if p.get("pairwise_rmse") is not None
    ]
    mean_icp_fitness = float(np.mean(fitnesses)) if fitnesses else None
    mean_icp_rmse = float(np.mean(rmses)) if rmses else None

    # Assembly bounding box from combined placed point clouds
    assembly_bbox_mm: Optional[List[float]] = None
    if placed_names:
        all_pts = []
        for name in placed_names:
            T = np.array(placements[name]["transform"])
            pts = np.asarray(frag_map[name].pcd.points)
            all_pts.append((T[:3, :3] @ pts.T + T[:3, 3:4]).T)
        stacked = np.vstack(all_pts)
        assembly_bbox_mm = (stacked.max(axis=0) - stacked.min(axis=0)).tolist()

    # Collision detection: check adjacent placed pairs from pairwise_results
    placed_set = set(placed_names)
    # Pre-compute transformed point arrays
    pts_global: Dict[str, np.ndarray] = {}
    for name in placed_names:
        T = np.array(placements[name]["transform"])
        pts = np.asarray(frag_map[name].pcd.points)
        pts_global[name] = (T[:3, :3] @ pts.T + T[:3, 3:4]).T

    max_collision = 0.0
    collision_flags: List[Dict[str, Any]] = []
    checked_pairs = set()

    for r in pairwise:
        fa, fb = r["fragment_a"], r["fragment_b"]
        if fa not in placed_set or fb not in placed_set:
            continue
        pair_key = tuple(sorted([fa, fb]))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        frac = _collision_fraction(pts_global[fa], pts_global[fb], collision_dist)
        max_collision = max(max_collision, frac)
        if frac > 0.05:
            collision_flags.append({
                "fragment_a": fa,
                "fragment_b": fb,
                "collision_fraction": round(frac, 4),
                "note": "possible misplacement" if frac > 0.20 else "minor contact",
            })

    return {
        "fragments_placed": n_placed,
        "total_fragments": n_total,
        "placement_rate": round(placement_rate, 4),
        "mean_icp_fitness": round(mean_icp_fitness, 4) if mean_icp_fitness is not None else None,
        "mean_icp_rmse_mm": round(mean_icp_rmse, 4) if mean_icp_rmse is not None else None,
        "assembly_bounding_box_mm": (
            [round(v, 2) for v in assembly_bbox_mm] if assembly_bbox_mm else None
        ),
        "max_collision_fraction": round(max_collision, 4),
        "collision_flags": collision_flags,
        "fitness_context": (
            "heritage_fragment — fitness 0.15-0.65 is expected with missing material; "
            "standard robotics thresholds (>0.8) do not apply"
        ),
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def summarize_results(
    fragments: List[ProcessedFragment],
    pairwise_results: List[Dict[str, Any]],
    refined: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full pipeline summary written to outputs/metrics/summary.json.

    Args:
        fragments: all processed fragments.
        pairwise_results: output of compute_pairwise_matches.
        refined: output of refine_assembly.
        config: pipeline config dict.

    Returns:
        Summary dict with global metrics and per-pair table.
    """
    assembly = refined.get("assembly", {})
    global_metrics = compute_global_metrics(fragments, assembly, config)

    pair_summary = [
        {
            "fragment_a": r["fragment_a"],
            "fragment_b": r["fragment_b"],
            "ransac_fitness": round(float(r.get("ransac_fitness", 0.0)), 4),
            "icp_fitness": round(float(r.get("icp_fitness", 0.0)), 4),
            "icp_rmse_mm": round(float(r.get("icp_rmse", 0.0)), 4),
            "overlap_score": round(float(r.get("overlap_score", 0.0)), 4),
        }
        for r in pairwise_results
    ]

    return {
        "global": global_metrics,
        "pairwise": pair_summary,
        "refinement_method": refined.get("method", "unknown"),
    }
