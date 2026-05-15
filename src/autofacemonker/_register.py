"""Core: MVMP landmarks → Procrustes → MeshMonk nonrigid."""

from importlib.resources import files

import numpy as np
import trimesh
import meshmonk
from mvmp import Facemarker

# 7 verified anatomical MediaPipe landmark → template vertex correspondences
DEFAULT_CORRESPONDENCES = [
    (0,   3572),  # nasion
    (4,   3589),  # nose tip
    (133, 2436),  # left eye inner
    (362, 4648),  # right eye inner
    (61,  2310),  # left mouth
    (291, 4849),  # right mouth
    (152, 3543),  # chin
]


def _default_template():
    return str(files("autofacemonker.data").joinpath("template.ply"))


class AutoFaceMonker:
    """Register a facial template onto target meshes using MVMP + MeshMonk.

    Parameters
    ----------
    template : str, pathlib.Path, trimesh.Trimesh, or None
        Template mesh. None (default) uses the bundled template.ply.
    correspondences : list of (lmk_idx, tpl_vert_id) or None
        Manual correspondences. None (default) uses built-in 7-keypoint set.
    num_iterations : int
        MeshMonk nonrigid iterations (default 120).
    """

    def __init__(self, template=None, correspondences=None, num_iterations=120):
        tpl_path = template if isinstance(template, (str, type(None))) else None
        if tpl_path is None:
            tpl_path = _default_template()
        self.template = trimesh.load(tpl_path) if tpl_path is not None else template

        self.correspondences = correspondences or DEFAULT_CORRESPONDENCES
        self.num_iterations = num_iterations
        self._marker = Facemarker()

        self._lmk_indices = np.array([c[0] for c in self.correspondences])
        self._tpl_vids = np.array([c[1] for c in self.correspondences])
        self._tmpl_lmks = self.template.vertices[self._tpl_vids]

    def register(self, target_path, save_path=None):
        """Register the template onto *target_path*.

        Parameters
        ----------
        target_path : str or pathlib.Path
            Path to the target .obj mesh.
        save_path : str, pathlib.Path, or None
            If provided, export the warped template as PLY.

        Returns
        -------
        warped_vertices : np.ndarray (N, 3)
        """
        target = trimesh.load(str(target_path), force="mesh")

        # ── MVMP ──────────────────────────────────────────────────────────
        res = self._marker.predict(str(target_path))
        lmks_full = np.full((478, 3), np.nan)
        for k, v in res.landmarks_3d.items():
            lmks_full[int(k)] = v

        # Keep only detected landmarks
        mask = ~np.isnan(lmks_full[self._lmk_indices, 0])
        idx = self._lmk_indices[mask]
        vid = self._tpl_vids[mask]
        tmpl = self._tmpl_lmks[mask]
        tgt = lmks_full[idx]

        # ── Procrustes (rotation + translation + scale) ───────────────────
        tmpl_c = tmpl - tmpl.mean(axis=0)
        tgt_c = tgt - tgt.mean(axis=0)
        H = tmpl_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        scale = np.sum(np.linalg.norm(tgt_c, axis=1)) / np.sum(
            np.linalg.norm(tmpl_c, axis=1)
        )
        t = tgt.mean(axis=0) - scale * R @ tmpl.mean(axis=0)

        aligned = scale * (R @ self.template.vertices.T).T + t
        aligned_n = (R @ self.template.vertex_normals.T).T

        resids = np.linalg.norm(aligned[self._tpl_vids[mask]] - tgt, axis=1)
        print(
            f"Procrustes: {mask.sum()} lmks  scale={scale:.0f}x  "
            f"P50={np.median(resids):.1f}mm  max={resids.max():.1f}mm"
        )

        # ── MeshMonk nonrigid ─────────────────────────────────────────────
        # MeshMonk defaults (sigma=3.0) assume mm-scale meshes (~100–300).
        # If vertex coords are all fractional (< 0.X) rather than 100+,
        # the mesh is in metres / another non-mm unit — upscale to ~200.
        max_coord = np.abs(target.vertices).max()
        scale_mm = 1.0
        if max_coord < 1.0:
            scale_mm = 200.0 / max_coord if max_coord > 0 else 1.0
            aligned *= scale_mm
            target.vertices = target.vertices * scale_mm

        tgt_features = np.column_stack([target.vertices, target.vertex_normals])
        result = meshmonk.nonrigid_register(
            floating_features=np.column_stack([aligned, aligned_n]),
            target_features=tgt_features,
            floating_faces=self.template.faces,
            target_faces=target.faces,
            num_iterations=self.num_iterations,
        )

        warped = result.aligned_vertices
        if scale_mm != 1.0:
            warped = warped / scale_mm

        if save_path:
            out = self.template.copy()
            out.vertices = warped
            out.export(str(save_path))
            print(f"Saved {save_path}")

        return warped


def register_template(target_path, template=None, correspondences=None, save_path=None):
    """Convenience function.  See ``AutoFaceMonker`` for parameters."""
    m = AutoFaceMonker(template=template, correspondences=correspondences)
    return m.register(target_path, save_path=save_path)
