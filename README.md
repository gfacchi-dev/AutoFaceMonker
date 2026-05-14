# AutoFaceMonker

Automatic 3D facial template registration using [MVMP](https://github.com/gfacchi-dev/mvmp) landmark detection and [MeshMonk](https://github.com/jsnyde0/meshmonk) nonrigid surface registration.

Given a template mesh and a target 3D face scan, AutoFaceMonker detects 478 MediaPipe facial landmarks via MVMP, aligns the template with Procrustes analysis, then refines the fit with MeshMonk nonrigid registration — no manual intervention required.

## Installation

```bash
pip install autofacemonker
```

Requires Python ≥ 3.11.

## Quick Start

```bash
autofacemonker subject.obj -o warped.ply
```

This uses the bundled template mesh and built-in 7-point anatomical landmark correspondences.

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
  -n, --iterations    MeshMonk nonrigid iterations (default: 120)
```

### Correspondence JSON format

```json
{"0": 3572, "4": 3589, "133": 2436, "362": 4648, "61": 2310, "291": 4849, "152": 3543}
```

## How It Works

1. **MVMP** detects 478 MediaPipe facial landmarks on the target mesh using multi-view 2D projections with 5 zone cameras.

2. **Procrustes** rigidly aligns the template using the 7 anatomical landmark correspondences, computing rotation, translation, and uniform scale.

3. **MeshMonk nonrigid** refines the fit by deforming the template to match the target surface over 120 iterations.

## Requirements

- Python ≥ 3.11
- meshmonk ≥ 0.3.0
- mvmp ≥ 1.3.0
- trimesh
- numpy

## License

MIT
