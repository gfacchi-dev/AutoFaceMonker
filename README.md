# AutoFaceMonker

Automatic 3D facial template registration using [MVMP](https://github.com/gfacchi-dev/mvmp) landmark detection and [MeshMonk](https://github.com/gfacchi-dev/meshmonk) nonrigid surface registration.

Given a template mesh and a target 3D face scan, AutoFaceMonker detects 478 MediaPipe facial landmarks via MVMP, aligns the template with Procrustes analysis, then refines the fit with MeshMonk nonrigid registration — no manual intervention required.

## Installation

```bash
pip install "autofacemonker @ git+https://github.com/gfacchi-dev/AutoFaceMonker.git@v0.3.0"
```

Requires Python ≥ 3.11. From 0.3.0 AutoFaceMonker depends on the
[gfacchi-dev/meshmonk](https://github.com/gfacchi-dev/meshmonk) fork (≥ 0.4.0), which adds
point-to-surface correspondences. Installing it from git builds meshmonk from source (CMake and
a C++20 compiler); prebuilt wheels are attached to the
[meshmonk releases](https://github.com/gfacchi-dev/meshmonk/releases). Releases up to 0.2.0 remain
on PyPI (`pip install autofacemonker`) with upstream meshmonk.

## Quick Start

```bash
autofacemonker subject.obj -o warped.ply
```

This uses the bundled template mesh and built-in 5-point anatomical landmark correspondences.

## Python API

```python
from autofacemonker import AutoFaceMonker

# Use default template and correspondences
monker = AutoFaceMonker()
warped_vertices = monker.register("subject.obj", save_path="warped.ply")
```

### Custom template and correspondences

```python
monker = AutoFaceMonker(
    template="my_template.ply",
    correspondences=[
        (0,   3572),   # nasion       → template vertex 3572
        (4,   3589),   # nose tip     → template vertex 3589
        (133, 2436),   # left eye     → template vertex 2436
        (362, 4648),   # right eye    → template vertex 4648
        (61,  2310),   # left mouth   → template vertex 2310
        (291, 4849),   # right mouth  → template vertex 4849
        (152, 3543),   # chin         → template vertex 3543
    ],
    num_iterations=200,
)
warped = monker.register("subject.obj")
```

## CLI

```
usage: autofacemonker <target.obj> [options]

positional arguments:
  target              Path to target .obj mesh

options:
  -t, --template      Template mesh path (default: bundled template.ply)
  -c, --correspondences
                      JSON file with landmark→vertex mapping
  -o, --out           Output PLY path (default: <target>_warped.ply)
  -n, --iterations    MeshMonk nonrigid iterations (default: 80)
  --point-to-surface  Point-to-surface correspondences (see below)
```

## Point-to-surface correspondences

```python
monker = AutoFaceMonker(point_to_surface=True)
```

By default MeshMonk matches each template vertex to a distance-weighted blend of nearby
*target vertices*, so re-tessellating the same surface moves the registration. With
`point_to_surface=True` each template vertex matches the closest point on the target *surface*,
which makes the fit independent of target sampling. On the AppValidation study (1016 scans from
5 facial scanners), re-meshing the targets moved the registered template by 0.63 mm on average
under the default rule — 1.04 mm on smartphone photogrammetry — and by 0.15 mm with
point-to-surface.

The rule only pushes the template toward the target, so it can slide off thin structures: on
LAFAS four ear landmarks moved by 3.5–9.9 mm. It is therefore off by default. Use it for
face-region analyses and device comparisons; keep the default when ears matter. It requires the
[gfacchi-dev/meshmonk](https://github.com/gfacchi-dev/meshmonk) fork, and `AutoFaceMonker` raises
`RuntimeError` on a meshmonk build without it instead of silently using the default rule
(`autofacemonker.meshmonk_supports_point_to_surface()` reports which build is loaded).

### Correspondence JSON format

```json
{"0": 3572, "4": 3589, "133": 2436, "362": 4648, "61": 2310, "291": 4849, "152": 3543}
```

## How It Works

1. **MVMP** detects 478 MediaPipe facial landmarks on the target mesh using multi-view 2D projections with 5 zone cameras.

2. **Procrustes** rigidly aligns the template using the 5 anatomical landmark correspondences (upper lip, both inner canthi, both mouth corners), computing rotation, translation, and uniform scale.

3. **MeshMonk nonrigid** refines the fit by deforming the template to match the target surface. The target is first cropped to faces within 12 mm of the aligned template, and the nonrigid registration uses parameters matching Cliniface's rNonRigid configuration (80 iterations, sigma 1.6, push-pull equalisation). If the crop would leave part of the template without target surface nearby (coverage below 98%), the margin is widened to 1.5× and then 2× rather than skipping the crop: an uncropped head lets neck, ears and hair drag the template outward.

## Requirements

- Python ≥ 3.11
- meshmonk ≥ 0.4.0 from [gfacchi-dev/meshmonk](https://github.com/gfacchi-dev/meshmonk)
- mvmp ≥ 1.4.2
- trimesh
- numpy
- scipy

## License

MIT
