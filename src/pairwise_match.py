"""Pairwise fragment registration: FPFH + RANSAC + ICP (Phases 2–4)."""

from __future__ import annotations

import logging
import time
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

from src.features import compute_fpfh, extract_matching_surface
from src.preprocess import ProcessedFragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ransac(
    src_down: o3d.geometry.PointCloud,
    tgt_down: o3d.geometry.PointCloud,
    src_fpfh: "o3d.pipelines.registration.Feature",
    tgt_fpfh: "o3d.pipelines.registration.Feature",
    config: Dict[str, Any],
) -> Optional["o3d.pipelines.registration.RegistrationResult"]:
    """Run RANSAC global registration on downsampled PCDs with FPFH features.

    Returns None if the result fitness is below min_fitness_to_keep, so that
    the caller can skip ICP for pairs with no geometric support.

    Args:
        src_down: downsampled source PCD with normals.
        tgt_down: downsampled target PCD with normals.
        src_fpfh: FPFH features for source (shape 33 x N_src).
        tgt_fpfh: FPFH features for target (shape 33 x N_tgt).
        config: pipeline config dict.

    Returns:
        RegistrationResult or None.
    """
    ransac_cfg = config["ransac"]
    max_corr = float(ransac_cfg["max_correspondence_distance"])
    max_iter = int(ransac_cfg["max_iterations"])
    confidence = float(ransac_cfg["confidence"])
    min_fitness = float(ransac_cfg["min_fitness_to_keep"])

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=False,
        max_correspondence_distance=max_corr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, confidence),
    )

    if result.fitness < min_fitness:
        return None
    return result


def _run_icp(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    config: Dict[str, Any],
) -> "o3d.pipelines.registration.RegistrationResult":
    """Refine a RANSAC transform using point-to-plane ICP.

    Uses max_correspondence_distance larger than RANSAC (design_v2 §G) so that
    points across missing-material gaps are not forced into false correspondences
    but are simply excluded from the minimisation.

    Args:
        src_pcd: full-resolution source PCD with normals.
        tgt_pcd: full-resolution target PCD with normals.
        init_transform: 4x4 initial transform from RANSAC.
        config: pipeline config dict.

    Returns:
        ICP RegistrationResult.
    """
    icp_cfg = config["icp"]
    max_corr = float(icp_cfg["max_correspondence_distance"])
    max_iter = int(icp_cfg["max_iterations"])

    return o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_corr,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )


def _pair_collision_fraction(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    transform: np.ndarray,
    collision_dist: float,
) -> float:
    """Compute the fraction of source points within collision_dist of target
    after applying the proposed transform.

    A real fracture match contacts only at the fracture face, so collision_fraction
    is at most the fraction of total surface that the fracture face occupies
    (typically 0.10–0.30). Interpenetrating false matches produce values >> 0.35.

    Args:
        src_pcd: source point cloud (untransformed).
        tgt_pcd: target point cloud (in its original frame).
        transform: 4x4 proposed transform (maps src to tgt frame).
        collision_dist: proximity threshold in mm.

    Returns:
        Fraction in [0, 1].
    """
    pts = np.asarray(src_pcd.points)
    pts_t = (transform[:3, :3] @ pts.T + transform[:3, 3:4]).T

    src_t = o3d.geometry.PointCloud()
    src_t.points = o3d.utility.Vector3dVector(pts_t)

    reg = o3d.pipelines.registration.evaluate_registration(
        src_t, tgt_pcd, collision_dist, np.eye(4),
    )
    return float(reg.fitness)


def _overlap_score(
    icp_fitness: float,
    icp_rmse: float,
    max_corr_dist: float,
    high_fitness_threshold: float = 0.65,
) -> float:
    """Compute overlap_score per design_v2 §C with a high-fitness penalty.

    Base formula:
        overlap_score = icp_fitness * max(0, 1 - icp_rmse / max_corr_dist)

    High-fitness penalty:
        For real heritage fragments, icp_fitness > 0.65 is suspiciously high —
        it suggests a false match (flat face aligned to flat face) rather than
        a true fracture match. A soft penalty down-ranks such pairs:
        if icp_fitness > high_fitness_threshold:
            score *= (1 - 0.5 * (icp_fitness - high_fitness_threshold))

    Args:
        icp_fitness: fraction of source points with correspondence <= max_corr_dist.
        icp_rmse: mean distance of matched correspondences (mm).
        max_corr_dist: ICP max_correspondence_distance (mm).
        high_fitness_threshold: icp_fitness above this is penalised (default 0.65).

    Returns:
        Scalar overlap_score in [0, 1].
    """
    score = float(icp_fitness * max(0.0, 1.0 - icp_rmse / max_corr_dist))
    if icp_fitness > high_fitness_threshold:
        excess = icp_fitness - high_fitness_threshold
        score *= max(0.0, 1.0 - 0.5 * excess)
    return score


# ---------------------------------------------------------------------------
# Per-pair registration
# ---------------------------------------------------------------------------

def register_pair(
    src: ProcessedFragment,
    tgt: ProcessedFragment,
    src_surface: o3d.geometry.PointCloud,
    tgt_surface: o3d.geometry.PointCloud,
    src_down: o3d.geometry.PointCloud,
    tgt_down: o3d.geometry.PointCloud,
    src_fpfh: "o3d.pipelines.registration.Feature",
    tgt_fpfh: "o3d.pipelines.registration.Feature",
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Register one (source, target) fragment pair via RANSAC then ICP.

    After ICP, a collision check is performed on the proposed transform.
    If the fraction of source points within collision_distance_mm of the target
    exceeds collision_reject_threshold, the pair is rejected as physically
    implausible (interpenetrating fragments).

    A real fracture match contacts only at the fracture face (collision_fraction
    typically 0.10–0.30). A false face-to-face interpenetration produces
    collision_fraction > 0.40.

    Args:
        src: source ProcessedFragment.
        tgt: target ProcessedFragment.
        src_surface / tgt_surface: matching surface PCDs (full resolution).
        src_down / tgt_down: downsampled PCDs used for RANSAC feature matching.
        src_fpfh / tgt_fpfh: FPFH features aligned with downsampled PCDs.
        config: pipeline config dict.

    Returns:
        Dict with transform, ransac_fitness, icp_fitness, icp_rmse,
        overlap_score, collision_fraction, is_near_duplicate;
        or None if rejected (below fitness threshold or implausible collision).
    """
    ransac_result = _run_ransac(src_down, tgt_down, src_fpfh, tgt_fpfh, config)
    if ransac_result is None:
        return None

    run_icp = bool(config["icp"].get("run_icp", True))
    if run_icp:
        icp_result = _run_icp(src_surface, tgt_surface, ransac_result.transformation, config)
        # Revert to RANSAC transform if ICP diverged (fitness dropped)
        if icp_result.fitness >= ransac_result.fitness:
            transform = icp_result.transformation
            icp_fitness = float(icp_result.fitness)
            icp_rmse = float(icp_result.inlier_rmse)
        else:
            logger.debug(
                "ICP diverged for (%s, %s): ransac_fit=%.3f icp_fit=%.3f — reverting to RANSAC",
                src.name, tgt.name, ransac_result.fitness, icp_result.fitness,
            )
            transform = ransac_result.transformation
            icp_fitness = float(ransac_result.fitness)
            icp_rmse = float(ransac_result.inlier_rmse)
    else:
        transform = ransac_result.transformation
        icp_fitness = float(ransac_result.fitness)
        icp_rmse = float(ransac_result.inlier_rmse)

    # --- Collision check: reject physically implausible interpenetration ---
    pairwise_cfg = config.get("pairwise", {})
    collision_dist = float(config["assembly"].get("collision_distance_mm", 2.0))
    collision_reject = float(pairwise_cfg.get("collision_reject_threshold", 0.35))
    near_dup_thresh = float(pairwise_cfg.get("near_duplicate_fitness_threshold", 0.85))

    collision_frac = _pair_collision_fraction(src_surface, tgt_surface, transform, collision_dist)

    is_near_duplicate = (icp_fitness >= near_dup_thresh) and (collision_frac >= 0.5)

    if collision_frac > collision_reject:
        label = "NEAR-DUPLICATE" if is_near_duplicate else "COLLISION-REJECTED"
        logger.info(
            "  %s  (%s, %s)  icp_fit=%.3f  collision=%.3f > %.2f",
            label, src.name[-8:], tgt.name[-8:],
            icp_fitness, collision_frac, collision_reject,
        )
        return None

    max_corr = float(config["icp"]["max_correspondence_distance"])
    score = _overlap_score(icp_fitness, icp_rmse, max_corr)

    return {
        "fragment_a": src.name,
        "fragment_b": tgt.name,
        "transform": transform.tolist(),
        "ransac_fitness": float(ransac_result.fitness),
        "icp_fitness": icp_fitness,
        "icp_rmse": icp_rmse,
        "overlap_score": score,
        "collision_fraction": collision_frac,
        "is_near_duplicate": is_near_duplicate,
    }


# ---------------------------------------------------------------------------
# Phase 4 orchestration
# ---------------------------------------------------------------------------

def compute_pairwise_matches(
    fragments: List[ProcessedFragment],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute FPFH + RANSAC + ICP pairwise matches for all fragment pairs.

    Pipeline per design_v2 Phase 2–4:
      1. Extract matching surface PCD (full or candidate mode).
      2. Voxel-downsample and compute FPFH per fragment.
      3. For each C(N,2) pair: RANSAC -> ICP -> overlap_score.
      4. Sort by overlap_score descending, keep top max_pairs_to_keep.

    Missing-material handling: min_fitness_to_keep is set low (0.1) because
    real heritage matches will have reduced fitness wherever material is absent.

    Args:
        fragments: list of ProcessedFragment from Phase 0+1.
        config: pipeline config dict.

    Returns:
        Sorted list of pairwise result dicts (descending overlap_score).
        Each dict contains: fragment_a, fragment_b, transform, ransac_fitness,
        icp_fitness, icp_rmse, overlap_score.
    """
    t0 = time.perf_counter()
    n = len(fragments)
    n_pairs = n * (n - 1) // 2
    logger.info("Phase 2-4: matching %d fragments (%d pairs)", n, n_pairs)

    # --- Phase 2+3: extract surfaces and compute FPFH ---
    surfaces: Dict[str, o3d.geometry.PointCloud] = {}
    downs: Dict[str, o3d.geometry.PointCloud] = {}
    fpfhs: Dict[str, "o3d.pipelines.registration.Feature"] = {}

    for frag in fragments:
        surface = extract_matching_surface(frag, config)
        down, fpfh = compute_fpfh(surface, config)
        surfaces[frag.name] = surface
        downs[frag.name] = down
        fpfhs[frag.name] = fpfh
        logger.info(
            "  FPFH  %-40s  surface=%5d pts  down=%4d pts",
            frag.name, len(surface.points), len(down.points),
        )

    # --- Phase 4: RANSAC + ICP for every pair ---
    results: List[Dict[str, Any]] = []
    for pair_idx, (i, j) in enumerate(combinations(range(n), 2)):
        fi, fj = fragments[i], fragments[j]
        t_pair = time.perf_counter()

        result = register_pair(
            fi, fj,
            surfaces[fi.name], surfaces[fj.name],
            downs[fi.name], downs[fj.name],
            fpfhs[fi.name], fpfhs[fj.name],
            config,
        )
        elapsed = time.perf_counter() - t_pair

        if result is not None:
            results.append(result)
            logger.info(
                "  [%3d/%3d]  %-20s x %-20s  fit=%.3f  rmse=%5.2f  score=%.3f  coll=%.3f  %.1fs",
                pair_idx + 1, n_pairs,
                fi.name.split("_FR_")[-1] if "_FR_" in fi.name else fi.name[-10:],
                fj.name.split("_FR_")[-1] if "_FR_" in fj.name else fj.name[-10:],
                result["icp_fitness"], result["icp_rmse"], result["overlap_score"],
                result["collision_fraction"], elapsed,
            )
        else:
            logger.debug(
                "  [%3d/%3d]  %-20s x %-20s  BELOW THRESHOLD  %.1fs",
                pair_idx + 1, n_pairs,
                fi.name[-10:], fj.name[-10:], elapsed,
            )

    results.sort(key=lambda x: x["overlap_score"], reverse=True)
    max_keep = int(config.get("pairwise", {}).get("max_pairs_to_keep", 30))
    results = results[:max_keep]

    logger.info(
        "Pairwise matching done: %d / %d pairs passed threshold  total=%.1fs",
        len(results), n_pairs, time.perf_counter() - t0,
    )
    return results
