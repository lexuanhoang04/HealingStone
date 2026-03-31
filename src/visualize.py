"""Assembly PLY export and diagnostic plots (Phase 9)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
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
# Qualitative colour palette for fragment-colored assembly
# (18 distinct colours; cycles for N > 18)
# ---------------------------------------------------------------------------
_PALETTE: List[List[float]] = [
    [0.902, 0.098, 0.294],  # red
    [0.235, 0.706, 0.294],  # green
    [0.000, 0.510, 0.784],  # blue
    [1.000, 0.882, 0.098],  # yellow
    [0.961, 0.510, 0.192],  # orange
    [0.569, 0.118, 0.706],  # purple
    [0.275, 0.941, 0.941],  # cyan
    [0.941, 0.196, 0.902],  # magenta
    [0.824, 0.961, 0.235],  # lime
    [0.980, 0.745, 0.745],  # pink
    [0.000, 0.502, 0.502],  # teal
    [0.902, 0.745, 0.000],  # gold
    [0.667, 0.000, 0.000],  # maroon
    [0.000, 0.502, 0.000],  # dark green
    [0.000, 0.000, 0.502],  # navy
    [0.502, 0.502, 0.000],  # olive
    [0.502, 0.000, 0.502],  # violet
    [0.200, 0.600, 0.800],  # steel blue
]


def _palette_color(idx: int) -> np.ndarray:
    """Return a unique RGB colour array for fragment index idx."""
    return np.array(_PALETTE[idx % len(_PALETTE)], dtype=float)


def _short_name(name: str) -> str:
    """Return a short display label for a fragment name."""
    # e.g. NAR_ST_43B_FR_01_F_01_R_02  ->  FR_01
    import re
    m = re.search(r"FR_\d+", name)
    return m.group() if m else name.split("_")[-1]


def _transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) point array.

    Args:
        pts: (N, 3) points in original scan frame.
        T: 4x4 row-major transform matrix.

    Returns:
        (N, 3) points in the assembly frame.
    """
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

def export_assembly_plys(
    fragments: List[ProcessedFragment],
    assembly: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Export two assembly PLY point clouds (fragment-colored and vertex-colored).

    assembly_fragment_colors.ply — each fragment a distinct solid colour;
                                    useful for debugging fragment boundaries.
    assembly_vertex_colors.ply   — original scan vertex colours preserved;
                                    useful for human inspection of the surface.

    Both are point clouds (not fused meshes) to avoid watertightness issues.
    Only placed fragments (status != "unplaced") are included.

    Args:
        fragments: all processed fragments.
        assembly: output of assemble_fragments or refine_assembly["assembly"].
        output_dir: directory to write PLY files (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    placements = assembly.get("placements", {})

    frag_pts_col: List[np.ndarray] = []   # (N, 3) for fragment-colored
    frag_col_col: List[np.ndarray] = []   # (N, 3) RGB per point
    vert_pts_col: List[np.ndarray] = []
    vert_col_col: List[np.ndarray] = []

    n_placed = 0
    for idx, frag in enumerate(fragments):
        p = placements.get(frag.name, {})
        T = p.get("transform")
        if T is None:
            continue

        T = np.array(T)
        pts = np.asarray(frag.pcd.points)
        pts_g = _transform_pts(pts, T)
        n = len(pts)

        # Fragment-colored: solid colour per fragment
        solid = np.tile(_palette_color(idx), (n, 1))
        frag_pts_col.append(pts_g)
        frag_col_col.append(solid)

        # Vertex-colored: original scan colours or neutral grey fallback
        if frag.has_colors and len(frag.pcd.colors) == n:
            vc = np.asarray(frag.pcd.colors)
        else:
            vc = np.full((n, 3), 0.7, dtype=float)
        vert_pts_col.append(pts_g)
        vert_col_col.append(vc)

        n_placed += 1

    if n_placed == 0:
        logger.warning("No placed fragments found; skipping PLY export.")
        return

    def _make_pcd(pts_list: List[np.ndarray], col_list: List[np.ndarray]) -> o3d.geometry.PointCloud:
        """Build an open3d PointCloud from stacked arrays."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(pts_list))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(np.vstack(col_list), 0.0, 1.0))
        return pcd

    frag_pcd = _make_pcd(frag_pts_col, frag_col_col)
    vert_pcd = _make_pcd(vert_pts_col, vert_col_col)

    frag_path = output_dir / "assembly_fragment_colors.ply"
    vert_path = output_dir / "assembly_vertex_colors.ply"
    o3d.io.write_point_cloud(str(frag_path), frag_pcd)
    o3d.io.write_point_cloud(str(vert_path), vert_pcd)

    logger.info(
        "Exported PLYs: %d fragments  %d total points -> %s",
        n_placed, len(frag_pcd.points), output_dir,
    )


# ---------------------------------------------------------------------------
# Assembly preview (2D projections)
# ---------------------------------------------------------------------------

def _make_assembly_preview(
    fragments: List[ProcessedFragment],
    assembly: Dict[str, Any],
    mode: str,
) -> Optional[plt.Figure]:
    """Create a 3-panel 2D scatter plot of the assembly (XY, XZ, YZ planes).

    Args:
        fragments: all processed fragments.
        assembly: assembly dict.
        mode: "fragment" for per-fragment palette colors, "vertex" for scan colors.

    Returns:
        matplotlib Figure, or None if no fragments are placed.
    """
    placements = assembly.get("placements", {})
    panel_info = [
        (0, 1, "X (mm)", "Y (mm)", "XY plane"),
        (0, 2, "X (mm)", "Z (mm)", "XZ plane"),
        (1, 2, "Y (mm)", "Z (mm)", "YZ plane"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    placed_any = False

    for idx, frag in enumerate(fragments):
        p = placements.get(frag.name, {})
        T = p.get("transform")
        if T is None:
            continue

        T = np.array(T)
        pts = np.asarray(frag.pcd.points)
        pts_g = _transform_pts(pts, T)
        n = len(pts)

        # Determine colours (N, 3) for this fragment
        if mode == "vertex" and frag.has_colors and len(frag.pcd.colors) == n:
            colors = np.asarray(frag.pcd.colors)
        else:
            colors = np.tile(_palette_color(idx), (n, 1))

        # Subsample to ~2000 points per fragment for speed
        step = max(1, n // 2000)
        pts_s = pts_g[::step]
        col_s = np.clip(colors[::step], 0.0, 1.0)

        for ax, (xi, yi, xl, yl, title) in zip(axes, panel_info):
            ax.scatter(pts_s[:, xi], pts_s[:, yi], c=col_s, s=0.3, alpha=0.4, linewidths=0)

        placed_any = True

    if not placed_any:
        plt.close(fig)
        return None

    for ax, (xi, yi, xl, yl, title) in zip(axes, panel_info):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"Assembly preview — {mode} colors", fontsize=13)
    fig.tight_layout()
    return fig


def save_assembly_previews(
    fragments: List[ProcessedFragment],
    assembly: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save fragment-colored and vertex-colored assembly preview PNGs.

    Args:
        fragments: all processed fragments.
        assembly: assembly dict.
        output_dir: directory to write PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, fname in [
        ("fragment", "assembly_preview_fragment_colors.png"),
        ("vertex",   "assembly_preview_vertex_colors.png"),
    ]:
        fig = _make_assembly_preview(fragments, assembly, mode)
        if fig is not None:
            fig.savefig(output_dir / fname, dpi=150)
            plt.close(fig)
            logger.info("Saved %s", fname)


# ---------------------------------------------------------------------------
# Assembly graph plot
# ---------------------------------------------------------------------------

def save_assembly_graph(
    fragments: List[ProcessedFragment],
    assembly: Dict[str, Any],
    pairwise_results: List[Dict[str, Any]],
    output_dir: Path,
    config: Dict[str, Any],
) -> None:
    """Save a NetworkX spring-layout plot of the assembly compatibility graph.

    Nodes are colour-coded by status (anchor=red, placed=green, unplaced=grey).
    Edge colours encode overlap_score (yellow->red = low->high).

    Args:
        fragments: all processed fragments.
        assembly: assembly dict with placements.
        pairwise_results: list of pairwise match dicts.
        output_dir: directory to write assembly_graph.png.
        config: pipeline config dict.
    """
    if not _HAS_NX:
        logger.warning("networkx not available; skipping assembly_graph.png")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    placements = assembly.get("placements", {})
    min_score = float(config["assembly"]["min_score_threshold"])

    G = nx.Graph()
    for frag in fragments:
        G.add_node(frag.name)

    for r in pairwise_results:
        score = float(r.get("overlap_score", 0.0))
        if score > min_score:
            G.add_edge(r["fragment_a"], r["fragment_b"], weight=score)

    labels = {n: _short_name(n) for n in G.nodes()}
    node_colors = []
    for n in G.nodes():
        st = placements.get(n, {}).get("status", "unplaced")
        if st == "anchor":
            node_colors.append("#e74c3c")
        elif st in ("placed", "sub_anchor"):
            node_colors.append("#2ecc71")
        else:
            node_colors.append("#95a5a6")

    edges = list(G.edges())
    edge_weights = np.array([G[u][v]["weight"] for u, v in edges]) if edges else np.array([])
    if len(edge_weights) > 1:
        ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
    else:
        ew_norm = np.ones(len(edge_weights))

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, alpha=0.9)
    if edges:
        cmap = plt.cm.YlOrRd
        edge_colors = [cmap(w) for w in ew_norm]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=2.5, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8, font_weight="bold")

    ax.set_title("Assembly compatibility graph\n(edge colour = overlap_score, yellow→red = low→high)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="anchor"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="placed"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=10, label="unplaced"),
        ],
        loc="upper left",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "assembly_graph.png", dpi=150)
    plt.close(fig)
    logger.info("Saved assembly_graph.png")


# ---------------------------------------------------------------------------
# Fragment size plot and pairwise heatmap
# ---------------------------------------------------------------------------

def _save_fragment_sizes(
    fragments: List[ProcessedFragment],
    output_dir: Path,
) -> None:
    """Bar chart of bounding-box volume proxy per fragment."""
    names = [_short_name(f.name) for f in fragments]
    sizes = [float(np.prod(f.extent)) for f in fragments]

    fig, ax = plt.subplots(figsize=(max(10, len(names)), 5))
    ax.bar(names, sizes, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Bounding-box volume proxy (mm³)")
    ax.set_title("Fragment size proxy (bbox volume)")
    fig.tight_layout()
    fig.savefig(output_dir / "fragment_sizes.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fragment_sizes.png")


def _save_pairwise_heatmap(
    pairwise_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """NxN overlap_score heatmap for all fragment pairs."""
    all_names = sorted(
        {r["fragment_a"] for r in pairwise_results}
        | {r["fragment_b"] for r in pairwise_results}
    )
    short_names = [_short_name(n) for n in all_names]
    idx = {n: i for i, n in enumerate(all_names)}
    N = len(all_names)
    matrix = np.zeros((N, N), dtype=float)

    for r in pairwise_results:
        i = idx[r["fragment_a"]]
        j = idx[r["fragment_b"]]
        score = float(r.get("overlap_score", r.get("score", 0.0)))
        matrix[i, j] = score
        matrix[j, i] = score

    fig, ax = plt.subplots(figsize=(max(8, N), max(7, N - 1)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="hot")
    fig.colorbar(im, ax=ax, label="overlap_score")
    ax.set_xticks(range(N))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title("Pairwise overlap_score heatmap\n(0.15–0.65 is good for heritage fragments)")
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved pairwise_heatmap.png")


# ---------------------------------------------------------------------------
# Top-level entry point called by run_pipeline.py
# ---------------------------------------------------------------------------

def save_basic_plots(
    fragments: List[ProcessedFragment],
    pairwise_results: List[Dict[str, Any]],
    config: Dict[str, Any],
    assembly: Optional[Dict[str, Any]] = None,
) -> None:
    """Save all configured diagnostic plots.

    Plots saved (controlled by config["plots"] flags):
      fragment_sizes.png        — bar chart of fragment volumes (always useful)
      pairwise_heatmap.png      — overlap_score NxN matrix
      assembly_graph.png        — MST connectivity graph
      assembly_preview_*.png    — 2D XY/XZ/YZ scatter previews

    Args:
        fragments: processed fragments.
        pairwise_results: list of pairwise match dicts.
        config: pipeline config dict.
        assembly: assembly dict; required for graph and preview plots.
    """
    output_dir = Path(config["paths"]["output_dir"]) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if config["plots"].get("save_fragment_size_plot", True) and fragments:
        _save_fragment_sizes(fragments, output_dir)

    if config["plots"].get("save_pairwise_heatmap", True) and pairwise_results:
        _save_pairwise_heatmap(pairwise_results, output_dir)

    if assembly is not None:
        if config["plots"].get("save_assembly_graph", True):
            save_assembly_graph(fragments, assembly, pairwise_results, output_dir, config)

        if config["plots"].get("save_assembly_preview", True):
            save_assembly_previews(fragments, assembly, output_dir)
