"""Main entry point for the Healing Stones reconstruction pipeline.

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --input_dir /path/to/fragments
    python run_pipeline.py --config config.yaml --no_cache
    python run_pipeline.py --config config.yaml --phases 0 1
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.io_utils import ensure_output_dirs, load_config, save_json, setup_logging
from src.preprocess import fragment_summary, load_and_preprocess_all
from src.pairwise_match import compute_pairwise_matches
from src.assembly import assemble_fragments
from src.refine import refine_assembly
from src.metrics import summarize_results
from src.visualize import export_assembly_plys, save_basic_plots

logger = logging.getLogger(__name__)


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
        help="Override config paths.output_dir.",
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


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides to the loaded config dict.

    CLI flags take precedence over config file values.

    Args:
        config: config dict loaded from YAML.
        args: parsed CLI arguments.

    Returns:
        Updated config dict.
    """
    if args.input_dir is not None:
        logger.info("CLI override: input_3d_dir = %s", args.input_dir)
        config["paths"]["input_3d_dir"] = args.input_dir
    if args.output_dir is not None:
        logger.info("CLI override: output_dir = %s", args.output_dir)
        config["paths"]["output_dir"] = args.output_dir
    return config


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

    # Save preprocessing summary
    output_dir = Path(config["paths"]["output_dir"])
    summary_rows = fragment_summary(fragments)
    save_json(summary_rows, output_dir / "metrics" / "fragments_summary.json")
    logger.info("Fragment summary saved to outputs/metrics/fragments_summary.json")

    # Log overview table
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


def main() -> None:
    """Run the full pipeline (or a subset of phases) from config."""
    t_pipeline_start = time.perf_counter()
    args = parse_args()

    # Load config and set up output dirs first (logging needs the log dir)
    config = load_config(args.config)
    if args.input_dir:
        config["paths"]["input_3d_dir"] = args.input_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    ensure_output_dirs(config)

    # Set up logging (file + console) after output dirs are created
    output_dir = Path(config["paths"]["output_dir"])
    setup_logging(config, output_dir / "logs")
    logger.info("Pipeline started. Config: %s", args.config)
    logger.info("Input dir:  %s", config["paths"]["input_3d_dir"])
    logger.info("Output dir: %s", config["paths"]["output_dir"])

    # Seed for reproducibility (numpy; Open3D RANSAC uses internal seed)
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    logger.info("Random seed: %d", seed)

    # Determine which phases to run
    run_all = args.phases is None
    phases = set(args.phases) if args.phases is not None else set()

    # --- Phase 0 + 1 ---
    fragments = []
    if run_all or (0 in phases) or (1 in phases):
        fragments = run_phase_0_and_1(config, force_decimate=args.no_cache)
        if not fragments:
            return

    if not run_all and not (phases - {0, 1}):
        elapsed = time.perf_counter() - t_pipeline_start
        logger.info("Requested phases complete. Total time: %.1fs", elapsed)
        return

    # --- Phase 2–4: pairwise FPFH + RANSAC + ICP ---
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

    # --- Phase 7: optional global ICP refinement ---
    logger.info("=" * 60)
    logger.info("PHASE 7: global refinement")
    logger.info("=" * 60)
    refined = refine_assembly(fragments, assembly, config)
    refined_assembly = refined.get("assembly", assembly)

    # --- Phase 8-9: metrics + export ---
    logger.info("=" * 60)
    logger.info("PHASE 8-9: metrics + export")
    logger.info("=" * 60)

    summary = summarize_results(fragments, pairwise_results, refined, config)
    save_json(summary, output_dir / "metrics" / "summary.json")
    logger.info("Metrics saved -> outputs/metrics/summary.json")

    save_json(pairwise_results, output_dir / "metrics" / "pairwise_scores.json")
    logger.info("Pairwise scores saved -> outputs/metrics/pairwise_scores.json")

    transforms_data = _build_transforms_json(refined_assembly)
    save_json(transforms_data, output_dir / "transforms.json")
    logger.info("Transforms saved -> outputs/transforms.json")

    export_assembly_plys(fragments, refined_assembly, output_dir / "meshes")

    save_basic_plots(fragments, pairwise_results, config, assembly=refined_assembly)

    elapsed = time.perf_counter() - t_pipeline_start
    logger.info("=" * 60)
    logger.info("Pipeline complete. Total time: %.1fs", elapsed)
    logger.info("Results in: %s", output_dir)

    # Print a quick summary to stdout
    g = summary.get("global", {})
    print(f"\n{'='*50}")
    print(f"  Fragments placed : {g.get('fragments_placed')} / {g.get('total_fragments')}")
    print(f"  Placement rate   : {g.get('placement_rate', 0):.0%}")
    print(f"  Mean ICP fitness : {g.get('mean_icp_fitness')}")
    print(f"  Mean ICP RMSE    : {g.get('mean_icp_rmse_mm')} mm")
    print(f"  Max collision    : {g.get('max_collision_fraction')}")
    print(f"  {g.get('fitness_context', '')}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
