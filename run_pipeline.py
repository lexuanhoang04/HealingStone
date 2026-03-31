"""Main entry point for the Healing Stones reconstruction pipeline.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --input_dir /path/to/fragments
    python run_pipeline.py --config config.yaml --no_cache
    python run_pipeline.py --config config.yaml --phases 0 1

Each run creates a self-contained timestamped directory under outputs/runs/
so that results from different parameter settings can be compared easily.
Set disable_run_timestamp: true in config.yaml to write to output_dir directly.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from src.io_utils import ensure_output_dirs, load_config, save_json, setup_logging
from src.preprocess import fragment_summary, load_and_preprocess_all
from src.pairwise_match import compute_pairwise_matches
from src.assembly import assemble_fragments
from src.refine import refine_assembly
from src.metrics import summarize_results
from src.visualize import export_assembly_plys, save_basic_plots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Healing Stones: fragment reconstruction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Override config paths.input_3d_dir.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override config paths.output_dir (base dir; run subdir is created inside).",
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Ignore cached decimated meshes and re-process from scratch.",
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", default=None,
        metavar="N",
        help=(
            "Run only the specified phases (0=load+decimate, 1=preprocess). "
            "Default: run all implemented phases."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def _make_run_dir(config: Dict[str, Any], timestamp: str) -> Path:
    """Create and return the timestamped run directory.

    If disable_run_timestamp is true, returns the base output_dir unchanged.

    Structure:
        outputs/runs/YYYY-MM-DD_HH-MM-SS/
            metrics/
            plots/
            meshes/
            logs/

    Args:
        config: pipeline config dict (reads paths.output_dir and
                disable_run_timestamp).
        timestamp: formatted timestamp string YYYY-MM-DD_HH-MM-SS.

    Returns:
        Path to the run output directory (already exists after this call).
    """
    base = Path(config["paths"]["output_dir"])
    if config.get("disable_run_timestamp", False):
        run_dir = base
    else:
        run_dir = base / "runs" / timestamp

    for subdir in ("metrics", "plots", "meshes", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    return run_dir


def _save_run_config(config: Dict[str, Any], run_dir: Path, config_path: str) -> None:
    """Copy the config file into the run directory for reproducibility.

    Args:
        config: resolved config dict (after CLI overrides).
        run_dir: timestamped run directory.
        config_path: original config file path (used as fallback source).
    """
    dest = run_dir / "config.yaml"
    # Write the resolved config (includes any CLI overrides) rather than
    # copying the raw file, so the saved config exactly matches what ran.
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Phase 0+1 helper
# ---------------------------------------------------------------------------

def run_phase_0_and_1(config: dict, force_decimate: bool) -> list:
    """Run Phase 0 (load + decimate) and Phase 1 (preprocess).

    Args:
        config: pipeline config dict.
        force_decimate: if True, ignore cache and re-decimate all meshes.

    Returns:
        List of ProcessedFragment objects.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0+1: load, decimate, preprocess")
    logger.info("=" * 60)
    fragments = load_and_preprocess_all(config, force_decimate=force_decimate)
    if not fragments:
        logger.error("No fragments produced by Phase 0+1. Aborting.")
        return []

    output_dir = Path(config["paths"]["output_dir"])
    summary_rows = fragment_summary(fragments)
    save_json(summary_rows, output_dir / "metrics" / "fragments_summary.json")

    logger.info("")
    logger.info("  %-40s  %8s  %8s  %8s  %6s  %s",
                "Fragment", "V_raw", "V_dec", "PCD_pts", "Colors", "Diag_mm")
    logger.info("  " + "-" * 90)
    for row in summary_rows:
        logger.info(
            "  %-40s  %8d  %8d  %8d  %-6s  %.1f",
            row["name"],
            row["num_vertices_raw"],
            row["num_vertices_decimated"],
            row["num_pcd_points"],
            str(row["has_colors"]),
            row["bbox_diagonal_mm"],
        )
    logger.info("")

    return fragments


# ---------------------------------------------------------------------------
# transforms.json builder
# ---------------------------------------------------------------------------

def _build_transforms_json(assembly: Dict[str, Any]) -> Dict[str, Any]:
    """Build the transforms.json structure per design_v2 §B.

    Args:
        assembly: output of assemble_fragments or refine_assembly["assembly"].

    Returns:
        Serialisable dict matching the transforms.json schema.
    """
    placements = assembly.get("placements", {})
    fragments_out: Dict[str, Any] = {}

    for name, p in placements.items():
        T = p.get("transform")
        entry: Dict[str, Any] = {
            "transform": (
                [list(map(float, row)) for row in np.array(T).tolist()]
                if T is not None else None
            ),
            "status": p.get("status", "unplaced"),
            "pairwise_fitness": p.get("pairwise_fitness"),
            "pairwise_rmse": p.get("pairwise_rmse"),
        }
        if p.get("placed_via"):
            entry["placed_via"] = p["placed_via"]
        if p.get("cross_validation_fitness") is not None:
            entry["cross_validation_fitness"] = p["cross_validation_fitness"]
        if p.get("reason"):
            entry["reason"] = p["reason"]
        fragments_out[name] = entry

    return {
        "assembly_method": assembly.get("method", "greedy_mst"),
        "anchor_fragment": assembly.get("anchor"),
        "fragments": fragments_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full pipeline (or a subset of phases) from config."""
    t_pipeline_start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_args()

    # --- Load config and apply CLI overrides ---
    config = load_config(args.config)
    if args.input_dir:
        config["paths"]["input_3d_dir"] = args.input_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir

    # --- Create timestamped run directory and redirect output_dir ---
    run_dir = _make_run_dir(config, timestamp)
    config["paths"]["output_dir"] = str(run_dir)

    # Also ensure processed_dir exists (shared across runs, not inside run_dir)
    Path(config["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)

    # --- Logging: file named with timestamp inside run dir ---
    log_filename = f"pipeline_{timestamp}.log"
    setup_logging(config, run_dir / "logs", log_filename=log_filename)

    logger.info("=" * 60)
    logger.info("Healing Stones reconstruction pipeline")
    logger.info("Run timestamp : %s", timestamp)
    logger.info("Run directory : %s", run_dir)
    logger.info("Config file   : %s", args.config)
    logger.info("Input dir     : %s", config["paths"]["input_3d_dir"])
    logger.info("=" * 60)

    # --- Save resolved config into run dir ---
    _save_run_config(config, run_dir, args.config)
    logger.info("Config snapshot saved -> %s/config.yaml", run_dir)

    # --- Seed ---
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    logger.info("Random seed: %d", seed)

    # --- Determine phases ---
    run_all = args.phases is None
    phases = set(args.phases) if args.phases is not None else set()

    # --- Phase 0+1 ---
    fragments = []
    if run_all or (0 in phases) or (1 in phases):
        fragments = run_phase_0_and_1(config, force_decimate=args.no_cache)
        if not fragments:
            return

    if not run_all and not (phases - {0, 1}):
        elapsed = time.perf_counter() - t_pipeline_start
        logger.info("Requested phases complete. Total time: %.1fs", elapsed)
        return

    # --- Phase 2–4: FPFH + RANSAC + ICP ---
    logger.info("=" * 60)
    logger.info("PHASE 2-4: pairwise FPFH + RANSAC + ICP matching")
    logger.info("=" * 60)
    pairwise_results = compute_pairwise_matches(fragments, config)
    logger.info("Pairwise results: %d pairs passed threshold", len(pairwise_results))

    # --- Phase 5+6: assembly ---
    logger.info("=" * 60)
    logger.info("PHASE 5-6: MST assembly + cross-validation")
    logger.info("=" * 60)
    assembly = assemble_fragments(fragments, pairwise_results, config)

    # --- Phase 7: optional global refinement ---
    logger.info("=" * 60)
    logger.info("PHASE 7: global refinement")
    logger.info("=" * 60)
    refined = refine_assembly(fragments, assembly, config)
    refined_assembly = refined.get("assembly", assembly)

    # --- Phase 8-9: metrics + export ---
    logger.info("=" * 60)
    logger.info("PHASE 8-9: metrics + export")
    logger.info("=" * 60)
    output_dir = run_dir

    summary = summarize_results(fragments, pairwise_results, refined, config)
    save_json(summary, output_dir / "metrics" / "summary.json")
    logger.info("Metrics  -> metrics/summary.json")

    save_json(pairwise_results, output_dir / "metrics" / "pairwise_scores.json")
    logger.info("Pairs    -> metrics/pairwise_scores.json")

    transforms_data = _build_transforms_json(refined_assembly)
    save_json(transforms_data, output_dir / "transforms.json")
    logger.info("Transforms -> transforms.json")

    export_assembly_plys(fragments, refined_assembly, output_dir / "meshes")
    save_basic_plots(fragments, pairwise_results, config, assembly=refined_assembly)

    elapsed = time.perf_counter() - t_pipeline_start

    # --- Experiment summary (key metrics, one file per run for easy comparison) ---
    g = summary.get("global", {})
    stele = g.get("stele_shape", {})
    experiment_summary = {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "elapsed_seconds": round(elapsed, 1),
        "fragments_placed": g.get("fragments_placed"),
        "total_fragments": g.get("total_fragments"),
        "placement_rate": g.get("placement_rate"),
        "mean_icp_fitness": g.get("mean_icp_fitness"),
        "mean_icp_rmse_mm": g.get("mean_icp_rmse_mm"),
        "max_collision_fraction": g.get("max_collision_fraction"),
        "n_implausible_pairs": g.get("n_implausible_pairs"),
        "is_physically_plausible": g.get("is_physically_plausible"),
        "n_disconnected_components": g.get("n_disconnected_components"),
        "low_support_placements": g.get("low_support_placements", []),
        "stele_shape": stele,
    }
    save_json(experiment_summary, output_dir / "experiment_summary.json")
    logger.info("Experiment summary -> experiment_summary.json")

    logger.info("=" * 60)
    logger.info("Pipeline complete. Total time: %.1fs", elapsed)
    logger.info("Run saved in: %s", run_dir)

    # --- Console summary ---
    print(f"\n{'='*55}")
    print(f"  Run directory    : {run_dir}")
    print(f"  Fragments placed : {g.get('fragments_placed')} / {g.get('total_fragments')}")
    print(f"  Placement rate   : {g.get('placement_rate', 0):.0%}")
    print(f"  Mean ICP fitness : {g.get('mean_icp_fitness')}")
    print(f"  Mean ICP RMSE    : {g.get('mean_icp_rmse_mm')} mm")
    print(f"  Max collision    : {g.get('max_collision_fraction')}")
    print(f"  Slab ratio       : {stele.get('slab_ratio')}  (< 0.5 = slab-like)")
    print(f"  Elongation ratio : {stele.get('elongation_ratio')}  (> 1.3 = tall)")
    print(f"  Is stele-like    : {stele.get('is_stele_like')}")
    print(f"  {g.get('fitness_context', '')}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
