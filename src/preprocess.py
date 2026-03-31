"""Mesh loading, decimation, caching, and preprocessing for Phase 0 and Phase 1."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import open3d as o3d

from src.io_utils import discover_mesh_files

logger = logging.getLogger(__name__)


@dataclass
class MeshRecord:
    """Raw fragment mesh loaded directly from disk, before any processing."""

    name: str
    path: Path
    mesh: o3d.geometry.TriangleMesh
    num_vertices_raw: int
    num_triangles_raw: int


@dataclass
class ProcessedFragment:
    """Decimated, cleaned mesh with a sampled point cloud, ready for feature extraction.

    Coordinates are kept in the original scan frame (not centred or normalised).
    This means transforms computed during assembly are directly interpretable
    as "apply T to original vertices to place in assembly frame".
    """

    name: str
    path: Path
    mesh: o3d.geometry.TriangleMesh   # decimated mesh in original coordinates
    pcd: o3d.geometry.PointCloud      # point cloud sampled from decimated mesh, with normals
    center: np.ndarray                # centroid of decimated mesh bbox, shape (3,)
    extent: np.ndarray                # axis-aligned bbox extent in scene units, shape (3,)
    bbox_diagonal: float              # Euclidean length of bbox diagonal in scene units
    has_colors: bool                  # whether vertex colours are present
    num_vertices_raw: int             # vertex count before decimation
    num_triangles_raw: int            # triangle count before decimation
    num_vertices: int                 # vertex count after decimation
    num_triangles: int                # triangle count after decimation


# ---------------------------------------------------------------------------
# Phase 0 — load and decimate
# ---------------------------------------------------------------------------

def load_raw_meshes(config: Dict[str, Any]) -> List[MeshRecord]:
    """Discover and load all supported mesh files from the configured input directory.

    Files are discovered by extension (case-insensitive). No naming conventions
    are assumed — any valid mesh file in the directory is loaded.

    Args:
        config: pipeline config dict.

    Returns:
        List of MeshRecord objects for all non-empty meshes found.
    """
    input_dir = Path(config["paths"]["input_3d_dir"])
    extensions: List[str] = config["mesh"]["supported_extensions"]
    paths = discover_mesh_files(input_dir, extensions)

    if not paths:
        logger.warning("No mesh files found in %s with extensions %s", input_dir, extensions)
        return []

    logger.info("Discovered %d mesh file(s) in %s", len(paths), input_dir)

    records: List[MeshRecord] = []
    for path in paths:
        t0 = time.perf_counter()
        mesh = o3d.io.read_triangle_mesh(str(path))
        elapsed = time.perf_counter() - t0

        if mesh.is_empty():
            logger.warning("Skipping empty mesh: %s", path.name)
            continue

        n_verts = len(mesh.vertices)
        n_tris = len(mesh.triangles)
        logger.info(
            "Loaded  %-42s  %9d verts  %9d tris  %.1fs",
            path.name, n_verts, n_tris, elapsed,
        )
        records.append(MeshRecord(
            name=path.stem,
            path=path,
            mesh=mesh,
            num_vertices_raw=n_verts,
            num_triangles_raw=n_tris,
        ))

    logger.info("Loaded %d/%d meshes successfully", len(records), len(paths))
    return records


def _cache_path(record: MeshRecord, processed_dir: Path) -> Path:
    """Return the expected cache file path for a given mesh record."""
    return processed_dir / f"{record.name}_decimated.ply"


def _cache_valid(record: MeshRecord, cache_file: Path) -> bool:
    """Return True if the cache file exists and is at least as new as the source."""
    if not cache_file.exists():
        return False
    return cache_file.stat().st_mtime >= record.path.stat().st_mtime


def decimate_and_cache(
    record: MeshRecord,
    config: Dict[str, Any],
    force: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Decimate a mesh to the configured triangle target and cache the result.

    Uses quadric decimation, which preserves vertex colours and surface shape
    better than voxel grid methods for structured 3D scanner output.

    If the mesh already has fewer triangles than the target, no decimation
    is applied (a copy is returned and cached as-is).

    Args:
        record: raw mesh record from load_raw_meshes.
        config: pipeline config dict.
        force: if True, ignore any cached file and re-decimate.

    Returns:
        Decimated TriangleMesh in original scan coordinates.
    """
    use_cache = bool(config["mesh"].get("use_cache", True))
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(record, processed_dir)

    # --- cache hit ---
    if use_cache and not force and _cache_valid(record, cache_file):
        cached = o3d.io.read_triangle_mesh(str(cache_file))
        if not cached.is_empty():
            logger.info(
                "Cache hit  %-40s  %d tris",
                record.name, len(cached.triangles),
            )
            return cached
        logger.warning("Cache file is empty or corrupt, re-decimating: %s", cache_file)

    # --- decimate ---
    n_tris = record.num_triangles_raw
    voxel_size = float(config["mesh"]["voxel_size"])
    # Small meshes whose triangle count is already within a reasonable range
    # for the chosen voxel_size do not need decimation. Heuristic: skip if
    # fewer than 50K triangles, as vertex clustering at 2 mm would not reduce
    # them further in a meaningful way.
    skip_threshold = 50_000
    if n_tris <= skip_threshold:
        logger.info(
            "No decimation needed  %-30s  %d tris",
            record.name, n_tris,
        )
        decimated = o3d.geometry.TriangleMesh(record.mesh)
    else:
        # Use vertex clustering (O(n), seconds) rather than quadric decimation
        # (O(n log n), minutes for >10M triangle meshes).
        # voxel_size drives geometric resolution consistently with FPFH parameters.
        voxel_size = float(config["mesh"]["voxel_size"])
        t0 = time.perf_counter()
        decimated = record.mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        elapsed = time.perf_counter() - t0
        n_out = len(decimated.triangles)
        ratio = n_out / max(n_tris, 1) * 100
        logger.info(
            "Decimated  %-35s  %d -> %d tris  (%.2f%%)  voxel=%.1fmm  %.1fs",
            record.name, n_tris, n_out, ratio, voxel_size, elapsed,
        )

    # --- cache write ---
    if use_cache:
        o3d.io.write_triangle_mesh(str(cache_file), decimated)
        logger.debug("Cached decimated mesh -> %s", cache_file)

    return decimated


# ---------------------------------------------------------------------------
# Phase 1 — preprocess
# ---------------------------------------------------------------------------

def _make_camera_loc(verts: np.ndarray) -> list:
    """Compute a virtual camera position above the mesh centroid.

    Used to orient computed normals outward (away from interior) in a way
    that is agnostic to the fragment's physical orientation. The camera is
    placed far above the centroid along the Z axis of the local bounding box,
    which gives a reasonable outward direction for most fragment poses.

    Args:
        verts: (N, 3) array of vertex positions in scene units.

    Returns:
        Camera position as a Python list [x, y, z].
    """
    centroid = verts.mean(axis=0)
    z_range = verts[:, 2].max() - verts[:, 2].min()
    offset = max(z_range * 10.0, 100.0)  # at least 100 units above
    return (centroid + np.array([0.0, 0.0, offset])).tolist()


def preprocess_fragment(
    record: MeshRecord,
    mesh: o3d.geometry.TriangleMesh,
    config: Dict[str, Any],
) -> ProcessedFragment:
    """Clean a decimated mesh, compute normals, and sample a surface point cloud.

    The mesh is NOT centred or normalised — original scan coordinates are
    preserved throughout. This ensures the transforms computed downstream
    are interpretable relative to the original file's coordinate system.

    Args:
        record: original mesh record (provides name, path, raw counts).
        mesh: decimated mesh returned by decimate_and_cache.
        config: pipeline config dict.

    Returns:
        ProcessedFragment with cleaned mesh, sampled PCD, and metadata.
    """
    mesh_cfg = config["mesh"]
    normals_radius = float(mesh_cfg["normals_radius"])
    normals_max_nn = int(mesh_cfg["normals_max_nn"])
    sample_points = int(mesh_cfg["sample_points"])
    sample_method: str = mesh_cfg.get("sample_method", "poisson_disk")

    # --- copy to avoid mutating the cached mesh ---
    mesh = o3d.geometry.TriangleMesh(mesh)

    # --- clean ---
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_triangles()

    has_colors = mesh.has_vertex_colors()

    # --- vertex normals on mesh ---
    # compute_vertex_normals uses face winding order for consistent orientation.
    # orient_normals_towards_camera_location is only available on PointCloud,
    # not TriangleMesh, so we do orientation only on the sampled PCD below.
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    camera_loc = _make_camera_loc(verts)

    # --- bounding box metadata (in original scene units) ---
    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=float)
    extent = np.asarray(bbox.get_extent(), dtype=float)
    bbox_diagonal = float(np.linalg.norm(extent))

    # --- sample surface point cloud ---
    n_actual = len(mesh.vertices)
    n_sample = min(sample_points, n_actual)

    if sample_method == "poisson_disk":
        try:
            pcd = mesh.sample_points_poisson_disk(number_of_points=n_sample)
        except Exception as exc:
            logger.warning(
                "%s: Poisson disk sampling failed (%s), falling back to uniform",
                record.name, exc,
            )
            pcd = mesh.sample_points_uniformly(number_of_points=n_sample)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=n_sample)

    # --- normals on point cloud ---
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normals_radius, max_nn=normals_max_nn
        )
    )
    pcd.orient_normals_towards_camera_location(camera_loc)

    n_tris = len(mesh.triangles)
    logger.info(
        "Preprocessed  %-35s  %d verts  %d tris  pcd=%d  colors=%-5s  diag=%.1f",
        record.name, n_actual, n_tris, len(pcd.points),
        str(has_colors), bbox_diagonal,
    )

    return ProcessedFragment(
        name=record.name,
        path=record.path,
        mesh=mesh,
        pcd=pcd,
        center=center,
        extent=extent,
        bbox_diagonal=bbox_diagonal,
        has_colors=has_colors,
        num_vertices_raw=record.num_vertices_raw,
        num_triangles_raw=record.num_triangles_raw,
        num_vertices=n_actual,
        num_triangles=n_tris,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_and_preprocess_all(
    config: Dict[str, Any],
    force_decimate: bool = False,
) -> List[ProcessedFragment]:
    """Run Phase 0 + Phase 1 end-to-end for all fragments.

    Discovery -> load -> decimate (with caching) -> clean -> normals -> sample PCD.

    Args:
        config: pipeline config dict.
        force_decimate: if True, ignore all cached decimated meshes.

    Returns:
        List of ProcessedFragment objects, one per valid mesh file found.
    """
    t_start = time.perf_counter()
    records = load_raw_meshes(config)
    if not records:
        logger.error("No meshes loaded. Check input_3d_dir in config.")
        return []

    fragments: List[ProcessedFragment] = []
    for i, record in enumerate(records):
        logger.info("--- Processing %d/%d: %s ---", i + 1, len(records), record.name)
        decimated = decimate_and_cache(record, config, force=force_decimate)
        fragment = preprocess_fragment(record, decimated, config)
        fragments.append(fragment)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Phase 0+1 complete: %d/%d fragments  total=%.1fs",
        len(fragments), len(records), elapsed,
    )
    return fragments


def fragment_summary(fragments: List[ProcessedFragment]) -> List[Dict[str, Any]]:
    """Return a list of serialisable dicts summarising each processed fragment.

    Useful for writing to summary.json or logging a quick overview.

    Args:
        fragments: list from load_and_preprocess_all.

    Returns:
        List of dicts with name, counts, bbox, and color flag.
    """
    rows = []
    for f in fragments:
        rows.append({
            "name": f.name,
            "num_vertices_raw": f.num_vertices_raw,
            "num_triangles_raw": f.num_triangles_raw,
            "num_vertices_decimated": f.num_vertices,
            "num_triangles_decimated": f.num_triangles,
            "num_pcd_points": len(f.pcd.points),
            "has_colors": f.has_colors,
            "bbox_extent_mm": f.extent.tolist(),
            "bbox_diagonal_mm": round(f.bbox_diagonal, 2),
            "center_mm": f.center.tolist(),
        })
    return rows
