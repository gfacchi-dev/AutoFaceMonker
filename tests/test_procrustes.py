"""Unit tests for Procrustes alignment math (no GPU / heavy deps required)."""
import numpy as np


# Replicated from _register.py so these tests are self-contained pure-math checks.
def _procrustes(tmpl_pts, tgt_pts):
    tmpl_c = tmpl_pts - tmpl_pts.mean(axis=0)
    tgt_c = tgt_pts - tgt_pts.mean(axis=0)
    H = tmpl_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = np.sum(np.linalg.norm(tgt_c, axis=1)) / np.sum(np.linalg.norm(tmpl_c, axis=1))
    t = tgt_pts.mean(axis=0) - scale * R @ tmpl_pts.mean(axis=0)
    return R, scale, t


def _apply(pts, R, scale, t):
    return scale * (R @ pts.T).T + t


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


RNG = np.random.default_rng(0)
PTS = RNG.random((7, 3)) * 50.0


def test_identity():
    R, scale, t = _procrustes(PTS, PTS)
    np.testing.assert_allclose(_apply(PTS, R, scale, t), PTS, atol=1e-10)


def test_pure_translation():
    shift = np.array([10.0, -5.0, 3.0])
    tgt = PTS + shift
    R, scale, t = _procrustes(PTS, tgt)
    np.testing.assert_allclose(_apply(PTS, R, scale, t), tgt, atol=1e-10)
    np.testing.assert_allclose(scale, 1.0, atol=1e-10)


def test_pure_scale():
    s = 3.5
    tgt = PTS * s
    R, scale, t = _procrustes(PTS, tgt)
    np.testing.assert_allclose(_apply(PTS, R, scale, t), tgt, atol=1e-8)
    np.testing.assert_allclose(scale, s, atol=1e-8)


def test_rotation_90deg_z():
    tgt = ((_rz(np.pi / 2)) @ PTS.T).T
    R, scale, t = _procrustes(PTS, tgt)
    np.testing.assert_allclose(_apply(PTS, R, scale, t), tgt, atol=1e-10)
    np.testing.assert_allclose(scale, 1.0, atol=1e-10)


def test_combined_transform():
    true_R = _rz(np.pi / 4)
    true_s = 2.0
    true_t = np.array([5.0, -3.0, 7.0])
    tgt = true_s * (true_R @ PTS.T).T + true_t
    R, scale, t = _procrustes(PTS, tgt)
    np.testing.assert_allclose(_apply(PTS, R, scale, t), tgt, atol=1e-8)
    np.testing.assert_allclose(scale, true_s, atol=1e-8)


def test_reflection_corrected():
    # SVD can produce a reflection (det=-1); verify the sign-flip fix works.
    true_R = _rz(np.pi / 3)
    # Force a reflection by flipping one axis on the target
    tgt = (true_R @ PTS.T).T
    tgt[:, 2] *= -1
    R, _scale, _t = _procrustes(PTS, tgt)
    assert np.linalg.det(R) > 0, "R must be a proper rotation (det +1)"
