"""CLI entry point for meshmonker."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Register a facial template onto a target mesh",
        usage="autofacemonker <target.obj> [options]",
    )
    parser.add_argument("target", help="Path to target .obj mesh")
    parser.add_argument("-t", "--template", default=None,
                        help="Template mesh path (default: bundled template.ply)")
    parser.add_argument("-c", "--correspondences", default=None,
                        help="JSON file with landmark->vertex mapping, e.g. {\"0\": 3572, ...}")
    parser.add_argument("-o", "--out", default=None,
                        help="Output PLY path (default: <target>_warped.ply)")
    parser.add_argument("-n", "--iterations", type=int, default=120,
                        help="MeshMonk nonrigid iterations (default: 120)")
    args = parser.parse_args()

    from ._register import AutoFaceMonker

    # Load correspondences from JSON if provided
    corr = None
    if args.correspondences:
        with open(args.correspondences) as f:
            raw = json.load(f)
        corr = [(int(k), v) for k, v in raw.items()]

    import os
    out = args.out or os.path.splitext(args.target)[0] + "_warped.ply"

    monker = AutoFaceMonker(
        template=args.template,
        correspondences=corr,
        num_iterations=args.iterations,
    )
    monker.register(args.target, save_path=out)


if __name__ == "__main__":
    main()
