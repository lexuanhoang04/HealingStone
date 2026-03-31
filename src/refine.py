"""Optional global ICP refinement of the assembled reconstruction (Phase 7)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
import open3d as o3d

from src.preprocess import ProcessedFragment

logger = logging.getLogger(__name__)


def refine_assembly(
    fragments: List[ProcessedFragment],
    assembly: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Optionally refine each placed fragment via ICP against the merged scene.

    If refine.run_refinement is False (default), returns the assembly unchanged.
    When enabled, each placed fragment's transform is refined by running ICP
    against the union of all other placed point clouds. This corrects drift
    accumulated from chaining transforms through a long MST path.

    Args:
        fragments: preprocessed fragments.
        assembly: output of assemble_fragments.
        config: pipeline config dict.

    Returns:
        Dict with keys: method, assembly. The assembly placements may have
        updated transforms if refinement ran.
    """
    run = bool(config.get("refine", {}).get("run_refinement", False))

    if not run:
        logger.info("Phase 7: global refinement disabled (refine.run_refinement: false)")
        return {
            "method": "no_refinement",
            "assembly": assembly,
        }

    t0 = time.perf_counter()
    logger.info("Phase 7: global ICP refinement")

    refine_cfg = config["refine"]
    max_corr = float(refine_cfg.get("max_correspondence_distance", 3.0))
    max_iter = int(refine_cfg.get("max_iterations", 30))

    placements = assembly["placements"]
    frag_map = {f.name: f for f in fragments}

    placed_names = [
        n for n, p in placements.items()
        if p.get("transform") is not None
        and p["status"] in ("placed", "anchor", "sub_anchor")
    ]

    if len(placed_names) < 2:
        logger.info("Too few placed fragments for global refinement; skipping.")
        return {"method": "no_refinement", "assembly": assembly}

    # Build merged scene point cloud (union of all placed PCDs in assembly frame)
    def _transform_pcd(name: str) -> o3d.geometry.PointCloud:
        T = placements[name]["transform"]
        pts = (T[:3, :3] @ np.asarray(frag_map[name].pcd.points).T + T[:3, 3:4]).T
        normals = (T[:3, :3] @ np.asarray(frag_map[name].pcd.normals).T).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd

    n_refined = 0
    for name in placed_names:
        if placements[name]["status"] in ("anchor", "sub_anchor"):
            continue  # anchor stays fixed

        # Scene = all placed fragments except self
        scene_pts = []
        scene_nrm = []
        for other in placed_names:
            if other == name:
                continue
            T = placements[other]["transform"]
            pts = (T[:3, :3] @ np.asarray(frag_map[other].pcd.points).T + T[:3, 3:4]).T
            nrm = (T[:3, :3] @ np.asarray(frag_map[other].pcd.normals).T).T
            scene_pts.append(pts)
            scene_nrm.append(nrm)

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(np.vstack(scene_pts))
        scene_pcd.normals = o3d.utility.Vector3dVector(np.vstack(scene_nrm))

        # Source PCD in current assembly frame
        src_pcd = _transform_pcd(name)

        result = o3d.pipelines.registration.registration_icp(
            src_pcd, scene_pcd,
            max_corr,
            np.eye(4),      # already in assembly frame; refine from identity
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )

        # Compose: new_global = refinement_delta @ old_global
        T_old = placements[name]["transform"]
        T_new = result.transformation @ T_old
        placements[name]["transform"] = T_new
        n_refined += 1

    logger.info(
        "Global refinement complete: %d fragments refined  %.1fs",
        n_refined, time.perf_counter() - t0,
    )
    return {
        "method": "global_icp",
        "assembly": assembly,
    }
