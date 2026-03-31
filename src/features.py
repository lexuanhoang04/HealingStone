"""Matching surface extraction (Phase 2) and FPFH feature computation (Phase 3)."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import open3d as o3d

from src.preprocess import ProcessedFragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy helper kept for any callers that still use it
# ---------------------------------------------------------------------------

def compute_fragment_features(fragment: ProcessedFragment) -> Dict[str, np.ndarray | float]:
    """Compute simple global geometric features for a fragment.

    Returns extent, normalised extent, volume proxy, and mesh counts.
    """
    extent = fragment.extent.astype(float)
    volume_proxy = float(np.prod(extent))
    norm_extent = extent / (np.linalg.norm(extent) + 1e-8)
    return {
        "extent": extent,
        "norm_extent": norm_extent,
        "volume_proxy": volume_proxy,
        "num_vertices": float(fragment.num_vertices),
        "num_triangles": float(fragment.num_triangles),
    }


# ---------------------------------------------------------------------------
# Phase 2 — Matching surface extraction
# ---------------------------------------------------------------------------

def extract_matching_surface(
    fragment: ProcessedFragment,
    config: Dict,
) -> o3d.geometry.PointCloud:
    """Extract the candidate matching surface PCD for a fragment.

    Two modes (controlled by config["matching_surface"]["mode"]):
      full_surface      — return the full preprocessed PCD as-is (default).
      candidate_surface — extract low-curvature (fracture-like) regions via
                          normal-variance and colour-uniformity analysis.

    Args:
        fragment: preprocessed fragment with PCD and normals.
        config: pipeline config dict.

    Returns:
        PointCloud in original scan coordinates representing the matching surface.
    """
    mode: str = config["matching_surface"]["mode"]
    if mode == "full_surface":
        return fragment.pcd
    return _extract_candidate_surface(fragment, config)


def _extract_candidate_surface(
    fragment: ProcessedFragment,
    config: Dict,
) -> o3d.geometry.PointCloud:
    """Extract flat/fracture-like regions via k-NN normal variance analysis.

    Vertex is flagged if its k-NN normal variance is below the p-th percentile
    of the fragment's distribution (low variance = locally flat = fracture
    candidate). A secondary colour-uniformity signal is used when colours exist.
    Flagged vertices are dilated by one hop to form contiguous patches.

    Falls back to the full surface if the flagged region is smaller than
    min_region_fraction of all vertices.

    Args:
        fragment: preprocessed fragment.
        config: pipeline config dict.

    Returns:
        Sub-PointCloud of candidate matching surface.
    """
    cfg = config["matching_surface"]
    nvar_pct = float(cfg.get("normal_variance_percentile", 25))
    col_pct = float(cfg.get("color_std_percentile", 30))
    min_frac = float(cfg.get("min_region_fraction", 0.05))
    k = 10

    pcd = fragment.pcd
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n = len(pts)

    if n == 0:
        return pcd

    has_colors = fragment.has_colors and len(pcd.colors) == n
    colors_arr = np.asarray(pcd.colors) if has_colors else None

    tree = o3d.geometry.KDTreeFlann(pcd)
    nvar = np.zeros(n, dtype=float)
    col_std_arr = np.zeros(n, dtype=float)

    for i in range(n):
        _, idx, _ = tree.search_knn_vector_3d(pts[i], k + 1)
        neighbours = np.asarray(idx)[1:]          # exclude self
        nvar[i] = float(np.sum(np.var(normals[neighbours], axis=0)))
        if has_colors:
            col_std_arr[i] = float(np.mean(np.std(colors_arr[neighbours], axis=0)))

    nvar_thresh = float(np.percentile(nvar, nvar_pct))
    flagged = nvar < nvar_thresh

    if has_colors:
        col_thresh = float(np.percentile(col_std_arr, col_pct))
        flagged &= col_std_arr < col_thresh

    # 1-hop dilation
    dilated = flagged.copy()
    for i in np.where(flagged)[0]:
        _, idx, _ = tree.search_knn_vector_3d(pts[i], k + 1)
        dilated[np.asarray(idx)[1:]] = True

    frac = float(dilated.sum()) / n
    if frac < min_frac:
        logger.warning(
            "%s: candidate surface too small (%.2f < min %.2f), falling back to full surface",
            fragment.name, frac, min_frac,
        )
        return pcd

    sub = pcd.select_by_index(np.where(dilated)[0].tolist())
    logger.debug(
        "%s: candidate surface %.1f%% of points (%d / %d)",
        fragment.name, frac * 100, int(dilated.sum()), n,
    )
    return sub


# ---------------------------------------------------------------------------
# Phase 3 — FPFH feature computation
# ---------------------------------------------------------------------------

def compute_fpfh(
    pcd: o3d.geometry.PointCloud,
    config: Dict,
) -> Tuple[o3d.geometry.PointCloud, "o3d.pipelines.registration.Feature"]:
    """Voxel-downsample a PCD, re-estimate normals, and compute FPFH features.

    Steps (per design_v2 Phase 3):
      1. Voxel downsample to fpfh.voxel_size (mm).
      2. Re-estimate normals on the downsampled cloud.
      3. Compute FPFH with fpfh.radius and fpfh.max_nn.

    The returned Feature data matrix has shape (33, N_down).

    Args:
        pcd: input point cloud (should already have normals, but normals are
             re-estimated on the downsampled version for consistency).
        config: pipeline config dict (reads config["fpfh"]).

    Returns:
        Tuple of (downsampled PointCloud with normals, FPFH Feature object).
    """
    fpfh_cfg = config["fpfh"]
    voxel_size = float(fpfh_cfg["voxel_size"])
    radius = float(fpfh_cfg["radius"])
    max_nn = int(fpfh_cfg["max_nn"])

    pcd_down = pcd.voxel_down_sample(voxel_size)

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
    )

    logger.debug(
        "FPFH: %d -> %d pts after voxel=%.1f mm  feature=(%d, %d)",
        len(pcd.points), len(pcd_down.points), voxel_size,
        fpfh.data.shape[0], fpfh.data.shape[1],
    )
    return pcd_down, fpfh
