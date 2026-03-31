# Healing Stones — 3D Fragment Reconstruction Pipeline

GSoC 2026 Test Submission.

---

## Setup

```bash
source heal/bin/activate
# or: pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py --config config.yaml
```

Runs end-to-end with no user interaction. Results are saved to `outputs/runs/<timestamp>/`.

Optional flags:

```bash
--input_dir /path/to/fragments   # any folder of .ply / .obj files
--no_cache                       # re-process from scratch
--output_dir /path/to/results
```

---

## Results

Run on 17 Naranjo Stele 43B fragments (~3.3 min on 8 cores).

| Metric | Value |
|---|---|
| Fragments placed | 17 / 17 |
| Disconnected components | 16 |
| Mean ICP fitness | 0.333 |
| Mean ICP RMSE | 1.19 mm |
| Max collision fraction | 0.238 |
| Physically plausible | No |

Only 1 pairwise match (FR-14 ↔ FR-15) exceeded the edge-weight threshold — 15 fragments remain as isolated sub-anchors at their original orientations. See the proposal for a full failure analysis and planned improvements.

### Plots

| | |
|---|---|
| ![Assembly preview](outputs/runs/2026-03-31_01-15-22/plots/assembly_preview_fragment_colors.png) | ![Pairwise heatmap](outputs/runs/2026-03-31_01-15-22/plots/pairwise_heatmap.png) |
| Assembly (fragment colors) | Pairwise overlap heatmap |
| ![Assembly graph](outputs/runs/2026-03-31_01-15-22/plots/assembly_graph.png) | ![Fragment sizes](outputs/runs/2026-03-31_01-15-22/plots/fragment_sizes.png) |
| Assembly graph | Fragment sizes |

### Visualizations

![Original Fragments](assets/original_fragments.gif)
![Preprocessed Fragments](assets/preprocessed_fragments.gif)
![Reconstruction](assets/reconstructed_assembly.gif)

---

## Outputs

Each run produces:

```
outputs/runs/<timestamp>/
├── transforms.json          # 4×4 rigid transform per fragment
├── config.yaml              # exact config used (reproducibility)
├── metrics/summary.json     # global + per-pair metrics
├── meshes/                  # colored PLY point clouds
└── plots/                   # assembly previews, heatmap, graph
```
