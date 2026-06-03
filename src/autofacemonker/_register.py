"""Core: MVMP landmarks → Procrustes → MeshMonk nonrigid."""

from importlib.resources import files
from typing import Callable

import numpy as np
import trimesh
import meshmonk
from mvmp import Facemarker

# Anatomical MediaPipe landmark → AFM template vertex correspondences, used to
# seed the rigid Procrustes alignment. Verified against the Cliniface
# anthropometric map and confirmed on real warped scans (each MediaPipe landmark
# lands within ~1 mm of the paired AFM vertex). NOTE: MediaPipe numbers in
# image-mirror, so its "133"/"61" are the SUBJECT'S RIGHT side.
# The nose tip is deliberately excluded — MediaPipe's pronasale is noisy — and
# isn't needed: the inner canthi sit ~28 mm behind the upper lip, so these five
# points already span enough depth to constrain the rigid fit.
DEFAULT_CORRESPONDENCES = [
    (0,   3572),  # upper lip           (labiale superius)
    (133, 2627),  # right inner canthus (endocanthion; MediaPipe-mirrored "left")
    (362, 4532),  # left inner canthus  (endocanthion)
    (61,  2179),  # right mouth corner  (cheilion)
    (291, 4980),  # left mouth corner   (cheilion)
]


# MeshMonk nonrigid parameters matching Cliniface's rNonRigid configuration
# (FaceTools MaskRegistration.cpp: NonRigidRegistration(80,3,0.9,true,10,true,
# 10,50,1.6,80,1,80,1)). meshmonk exposes all of these except the smoothing
# neighbour count (smoothK=50 there, hardcoded here). Notably softer-but-more-
# local than the meshmonk defaults: sigma 1.6 (vs 3.0), flag threshold 0.9 (vs
# 0.999), push-pull equalisation on, kappa 10 (vs 12).
CLINIFACE_NONRIGID = {
    "transform_sigma": 1.6,
    "transform_num_viscous_iterations_start": 80,
    "transform_num_viscous_iterations_end": 1,
    "transform_num_elastic_iterations_start": 80,
    "transform_num_elastic_iterations_end": 1,
    "inlier_kappa": 10.0,
    "correspondences_flag_threshold": 0.9,
    "correspondences_equalize_push_pull": True,
}


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
        MeshMonk nonrigid update iterations (default 80, matching Cliniface).
    nonrigid_params : dict or None
        Extra MeshMonk ``nonrigid_register`` kwargs (sigma, viscous/elastic
        schedule, kappa, flag threshold, push-pull …). None (default) uses
        ``CLINIFACE_NONRIGID`` — the rNonRigid configuration Cliniface uses.
        Pass ``{}`` to fall back to plain meshmonk defaults.
    crop_margin : float or None
        Before the nonrigid step, crop the target to faces within this many mm
        of the rigidly-aligned template (as Cliniface does) so the morph isn't
        pulled toward neck/ears/hair. None disables cropping. Default 12.0.
    """

    def __init__(self, template=None, correspondences=None, num_iterations=80,
                 nonrigid_params=None, crop_margin=12.0):
        tpl_path = template if isinstance(template, (str, type(None))) else None
        if tpl_path is None:
            tpl_path = _default_template()
        self.template = trimesh.load(tpl_path) if tpl_path is not None else template

        self.correspondences = correspondences or DEFAULT_CORRESPONDENCES
        self.num_iterations = num_iterations
        self.nonrigid_params = (
            dict(CLINIFACE_NONRIGID) if nonrigid_params is None else dict(nonrigid_params)
        )
        self.crop_margin = crop_margin
        self._marker = Facemarker()

        self._lmk_indices = np.array([c[0] for c in self.correspondences])
        self._tpl_vids = np.array([c[1] for c in self.correspondences])
        self._tmpl_lmks = self.template.vertices[self._tpl_vids]

    def register(self, target_path, save_path=None,
                 progress_callback: Callable[[float, str], None] | None = None):
        """Register the template onto *target_path*.

        Parameters
        ----------
        target_path : str or pathlib.Path
            Path to the target .obj mesh.
        save_path : str, pathlib.Path, or None
            If provided, export the warped template as PLY.
        progress_callback : Callable[[float, str], None] or None
            Optional callback for progress updates. Called with (0.0–1.0, stage).

        Returns
        -------
        warped_vertices : np.ndarray (N, 3)
        """
        cb = progress_callback
        target = trimesh.load(str(target_path), force="mesh")

        # ── MVMP ──────────────────────────────────────────────────────────
        if cb: cb(0.0, "Detecting landmarks")
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
        if cb: cb(0.2, "Aligning template")
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

        # ── Target crop ───────────────────────────────────────────────────
        # Keep only target faces near the rigidly-aligned mask, so the nonrigid
        # step (with a local sigma) isn't pulled toward neck/ears/hair on full
        # head scans. Matches Cliniface's MaskRegistration crop step.
        tgt_v, tgt_f, tgt_n = target.vertices, target.faces, target.vertex_normals
        if self.crop_margin is not None:
            from scipy.spatial import cKDTree

            margin = self.crop_margin * scale_mm
            d, _ = cKDTree(aligned).query(target.vertices)
            keep_v = d < margin
            keep_f = keep_v[target.faces].all(axis=1)
            if keep_f.sum() > len(self.template.faces):  # guard against over-crop
                sub = target.submesh([np.where(keep_f)[0]], append=True)
                tgt_v, tgt_f, tgt_n = sub.vertices, sub.faces, sub.vertex_normals
                print(f"Cropped target to {len(tgt_v)} verts within {self.crop_margin}mm of mask")

        tgt_features = np.column_stack([tgt_v, tgt_n])

        if cb: cb(0.3, "Registering")
        result = meshmonk.nonrigid_register(
            floating_features=np.column_stack([aligned, aligned_n]),
            target_features=tgt_features,
            floating_faces=self.template.faces,
            target_faces=tgt_f,
            num_iterations=self.num_iterations,
            **self.nonrigid_params,
        )
        if cb: cb(0.95, "Completed")

        warped = result.aligned_vertices
        if scale_mm != 1.0:
            warped = warped / scale_mm

        if save_path:
            out = self.template.copy()
            out.vertices = warped
            out.export(str(save_path))
            print(f"Saved {save_path}")

        return warped


def register_template(target_path, template=None, correspondences=None,
                      save_path=None, progress_callback=None):
    """Convenience function.  See ``AutoFaceMonker`` for parameters."""
    m = AutoFaceMonker(template=template, correspondences=correspondences)
    return m.register(target_path, save_path=save_path,
                      progress_callback=progress_callback)
