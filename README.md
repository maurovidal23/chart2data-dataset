# Chart2Data Dataset Generator

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Chart2Data creates synthetic chart images together with exact numerical and pixel-space ground
truth. It is intended for training and evaluating systems that recover the data behind charts in
web pages, papers, dashboards, screenshots, and PDFs.

The repository is a dataset generator, not a model-training framework. It deliberately avoids
dependencies on PyTorch, Transformers, or any particular model architecture.

## Why this project exists

A chart-to-data model must do more than trace a line. It must locate the plot, read the axes,
understand their scale, and convert visual geometry back into numerical values. Training against
only one final representation loses information that is useful for other approaches.

Chart2Data therefore stores a renderer-independent canonical record for every image:

```text
synthetic series + axis plan + visual style
                    │
                    ▼
              chart renderer ─────► PNG
                    │
                    ├── original x/y values
                    ├── actual tick values and displayed labels
                    ├── plot bounding box
                    ├── data ↔ image-pixel transforms
                    ├── rendered point and curve coordinates
                    ├── normalized curve targets
                    └── style, backend, and signal metadata
```

Model-specific targets are derived from this record without changing the source truth.

## Highlights

- Deterministic synthetic time series with trends, seasonality, noise, anomalies, level shifts,
  and slope changes.
- Regular and irregular numeric x coordinates.
- Multiple visual renderings of the same latent series.
- Matplotlib and optional Plotly/Kaleido rendering backends.
- Deterministic backend selection independently for each rendering.
- Exact plot geometry and invertible data-to-pixel transforms.
- Leakage-safe train, validation, and test splits at latent-series level.
- JSONL and nested Parquet metadata.
- Raw-point, normalized-curve, pixel-curve, axis, and VLM JSON targets.
- Axis-scale challenge pairs with matching shapes but different numerical calibration.
- Reproducibility manifests and dataset validation tools.

Version 0.1 supports linear numeric axes. The schema reserves log, date, and categorical scale
types for later versions.

## Quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/maurovidal23/chart2data-dataset.git
cd chart2data-dataset
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Generate a small dataset:

```bash
chart2data generate \
  --config configs/dataset_debug.yaml \
  --output data/debug
```

Validate it and create a diagnostic overlay:

```bash
chart2data validate --dataset data/debug
chart2data inspect --dataset data/debug --sample-id 00000000
```

All generated output lives under `data/`, which is intentionally excluded from version control.

## Rendering backends

Matplotlib is installed by default. To generate visually distinct Plotly/Kaleido charts, install
the optional extra:

```bash
python -m pip install -e ".[dev,plotly]"
```

Choose a renderer in YAML:

```yaml
rendering:
  backend: auto  # matplotlib | plotly | auto
```

- `matplotlib` preserves the original default behavior.
- `plotly` always uses Plotly with Kaleido for static PNG export.
- `auto` selects a backend deterministically for each rendering using a separate random stream.

The separate backend-selection stream means that backend choice does not perturb signal, style,
or axis randomness. Related axis-challenge samples also receive matching renderer choices.

Kaleido 1.x requires Chrome. If Chrome is unavailable, install it separately or explicitly run
`kaleido_get_chrome`. That command downloads a browser and should only be run when network access
and the download are acceptable. Missing rendering dependencies produce an actionable error;
Chart2Data never silently falls back to another backend.

See [configs/dataset_plotly.yaml](configs/dataset_plotly.yaml) for a mixed-backend example.

## Dataset layout

```text
data/debug/
├── images/
│   ├── train/*.png
│   ├── validation/*.png
│   └── test/*.png
├── jsonl/{train,validation,test}.jsonl
├── metadata/{train,validation,test}.parquet
└── manifest.json
```

Each sample contains:

- identifiers, split, and image path;
- full-precision source points;
- axis limits, ticks, tick labels, formatter, and labels;
- image dimensions and plot bounds;
- forward and inverse homogeneous transform matrices;
- source points expressed in image pixels;
- a fixed-grid interpolated curve;
- plot-normalized and pixel-space curve representations;
- visual-resolution and renderer-style metadata;
- optional signal-component and challenge-group metadata.

The manifest records the configuration, stable configuration hash, generator version, split
counts, and renderer package versions.

## Coordinate conventions

All stored pixel coordinates use the raster-image convention:

- origin at the top-left of the complete PNG;
- x increases to the right;
- y increases downward.

The plot bounding box uses the same coordinate system. Both renderers return forward and inverse
3×3 affine matrices, allowing any point to be converted between numerical data coordinates and
PNG pixels.

Normalized pixel curves are plot-relative: `(0, 0)` is the plot's top-left and `(1, 1)` is its
bottom-right. Normalized numerical curves use `z=0` at the axis bottom and `z=1` at the axis top.

## Configuration

Dataset generation is controlled by YAML. The main options include:

```yaml
dataset_version: 0.1.0
global_seed: 42
num_latent_series: 100
renderings_per_series: 2
curve_grid_size: 128
num_workers: 1
save_components: true

signal:
  min_points: 32
  max_points: 128
  irregular_x_probability: 0.25

rendering:
  backend: matplotlib
  widths_px: [640, 800, 1024]
  heights_px: [480, 600, 768]
  dpis: [80, 100, 120]

split_ratios:
  train: 0.8
  validation: 0.1
  test: 0.1
```

The CLI can override common generation values:

```text
--num-series N
--renderings-per-series N
--seed N
--num-workers N
--curve-grid-size K
--save-components / --no-save-components
--overwrite
```

Random streams are derived from stable semantic keys. Output is independent of task ordering and
worker count when the same package/runtime versions are used.

## Derived targets

Canonical records can be converted into different supervision formats:

```python
from chart2data.targets import (
    build_axis_target,
    build_normalized_curve_target,
    build_pixel_curve_target,
    build_raw_points_target,
    build_vlm_json_target,
)

raw_points = build_raw_points_target(sample)
normalized_curve = build_normalized_curve_target(sample)
pixel_curve = build_pixel_curve_target(sample)
axis = build_axis_target(sample)
vlm_json = build_vlm_json_target(sample)
```

This supports sequence generation, numerical regression, pixel-space supervision, and hybrid
training objectives without coupling stored truth to one model output grammar.

## Axis-scale challenges

Axis-use challenges create paired series with the same normalized shape and rendering randomness
but numerical values separated by a deterministic scale factor. The pair shares a split and a
`challenge_group_id`, making it useful for measuring whether a model genuinely reads axis labels
instead of predicting only the visual curve shape.

Enable them in YAML:

```yaml
axis_challenges:
  enabled: true
  fraction: 0.2
  scale_factors: [100.0, 1000.0, 1000000.0]
```

## Development and verification

```bash
python -m pytest -q
python -m ruff check .
```

The test suite covers deterministic generation, metadata serialization, coordinate round trips,
pixel geometry, interpolation, normalized targets, split isolation, axis challenges, backend
selection, multiprocessing reproducibility, optional-dependency errors, and Plotly pixel probes.

## Project scope

Chart2Data currently generates single-series line charts with linear numeric axes. Useful future
extensions include additional chart types, logarithmic/date/categorical axes, more rendering
engines, OCR-specific challenges, and controlled image degradation.

Contributions that preserve deterministic generation and the canonical renderer contract are
welcome.

## License

Released under the [MIT License](LICENSE).
