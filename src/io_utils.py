"""Utilities for config loading, filesystem I/O, mesh file discovery, and logging."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """Create standard output directories if they do not exist."""
    output_dir = Path(config["paths"]["output_dir"])
    for subdir in ["logs", "plots", "meshes", "metrics"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str | Path) -> None:
    """Save a value as a formatted JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load and return the contents of a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_mesh_files(input_dir: str | Path, extensions: List[str]) -> List[Path]:
    """Return a sorted list of mesh file paths matching the given extensions.

    Matching is case-insensitive. Searches only the top-level directory
    (non-recursive) to avoid picking up cached or temporary files.

    Args:
        input_dir: directory to search.
        extensions: list of extensions including the dot, e.g. [".ply", ".obj"].

    Returns:
        Sorted list of Path objects for matched files.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logging.getLogger(__name__).error("Input directory does not exist: %s", input_dir)
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)


def setup_logging(
    config: Dict[str, Any],
    log_dir: str | Path,
    log_filename: str = "pipeline.log",
) -> None:
    """Configure the root logger with a console handler and a file handler.

    Calling this multiple times is safe: existing handlers are cleared first.

    Args:
        config: pipeline config dict (reads config["logging"]["level"]).
        log_dir: directory where the log file is written.
        log_filename: name of the log file (default "pipeline.log").
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = config.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    fh = logging.FileHandler(log_dir / log_filename, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)
