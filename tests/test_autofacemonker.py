"""Integration tests for AutoFaceMonker with mocked MVMP and MeshMonk."""
import numpy as np
import pytest
import trimesh
from unittest.mock import MagicMock, patch

from autofacemonker import AutoFaceMonker
from autofacemonker._register import DEFAULT_CORRESPONDENCES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def target_mesh(tmp_path):
    """Small but valid mesh saved to a real .obj file."""
    mesh = trimesh.creation.icosphere(subdivisions=4)  # 2562 verts
    path = tmp_path / "target.obj"
    mesh.export(str(path))
    return str(path)


@pytest.fixture
def monker():
    """AutoFaceMonker with Facemarker mocked out (uses real bundled template)."""
    with patch("autofacemonker._register.Facemarker") as mock_cls:
        mock_cls.return_value = MagicMock()
        m = AutoFaceMonker()
    # keep the mock instance accessible via m._marker
    return m


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

@patch("autofacemonker._register.Facemarker")
def test_init_loads_bundled_template(_mock_facemarker):
    monker = AutoFaceMonker()
    assert monker.template is not None
    assert monker.template.vertices.ndim == 2
    assert monker.template.vertices.shape[1] == 3


@patch("autofacemonker._register.Facemarker")
def test_init_default_correspondences(_mock_facemarker):
    monker = AutoFaceMonker()
    assert monker.correspondences == DEFAULT_CORRESPONDENCES
    assert len(monker._lmk_indices) == len(DEFAULT_CORRESPONDENCES)
    assert len(monker._tpl_vids) == len(DEFAULT_CORRESPONDENCES)


@patch("autofacemonker._register.Facemarker")
def test_init_custom_template(_mock_facemarker, tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=5)  # 10242 verts
    path = tmp_path / "custom.ply"
    mesh.export(str(path))
    monker = AutoFaceMonker(template=str(path))
    assert monker.template.vertices.shape == mesh.vertices.shape


@patch("autofacemonker._register.Facemarker")
def test_init_custom_correspondences(_mock_facemarker):
    custom = [(0, 100), (4, 200)]
    monker = AutoFaceMonker(correspondences=custom)
    assert monker.correspondences == custom


# ---------------------------------------------------------------------------
# register() tests
# ---------------------------------------------------------------------------

def _make_landmark_mock(monker):
    """Return a mock Predict result with all 7 default landmarks present."""
    lmk_indices = [c[0] for c in DEFAULT_CORRESPONDENCES]
    tpl_vids = [c[1] for c in DEFAULT_CORRESPONDENCES]
    # Use the actual template vertex positions — perfect alignment case.
    lmks_3d = {
        str(idx): monker.template.vertices[vid].tolist()
        for idx, vid in zip(lmk_indices, tpl_vids)
    }
    mock_pred = MagicMock()
    mock_pred.landmarks_3d = lmks_3d
    return mock_pred


@patch("autofacemonker._register.Facemarker")
@patch("autofacemonker._register.meshmonk")
def test_register_returns_correct_shape(mock_meshmonk, mock_facemarker_cls, target_mesh):
    mock_marker = MagicMock()
    mock_facemarker_cls.return_value = mock_marker

    monker = AutoFaceMonker()
    n_tpl = len(monker.template.vertices)

    mock_marker.predict.return_value = _make_landmark_mock(monker)

    mock_nr = MagicMock()
    mock_nr.aligned_vertices = np.zeros((n_tpl, 3))
    mock_meshmonk.nonrigid_register.return_value = mock_nr

    result = monker.register(target_mesh)

    assert result.shape == (n_tpl, 3)


@patch("autofacemonker._register.Facemarker")
@patch("autofacemonker._register.meshmonk")
def test_register_calls_marker_and_meshmonk(mock_meshmonk, mock_facemarker_cls, target_mesh):
    mock_marker = MagicMock()
    mock_facemarker_cls.return_value = mock_marker

    monker = AutoFaceMonker()
    n_tpl = len(monker.template.vertices)

    mock_marker.predict.return_value = _make_landmark_mock(monker)
    mock_nr = MagicMock()
    mock_nr.aligned_vertices = np.zeros((n_tpl, 3))
    mock_meshmonk.nonrigid_register.return_value = mock_nr

    monker.register(target_mesh)

    mock_marker.predict.assert_called_once_with(target_mesh)
    mock_meshmonk.nonrigid_register.assert_called_once()


@patch("autofacemonker._register.Facemarker")
@patch("autofacemonker._register.meshmonk")
def test_register_saves_ply(mock_meshmonk, mock_facemarker_cls, target_mesh, tmp_path):
    mock_marker = MagicMock()
    mock_facemarker_cls.return_value = mock_marker

    monker = AutoFaceMonker()
    n_tpl = len(monker.template.vertices)

    mock_marker.predict.return_value = _make_landmark_mock(monker)
    mock_nr = MagicMock()
    mock_nr.aligned_vertices = monker.template.vertices.copy()
    mock_meshmonk.nonrigid_register.return_value = mock_nr

    out_path = tmp_path / "warped.ply"
    monker.register(target_mesh, save_path=str(out_path))

    assert out_path.exists()
    saved = trimesh.load(str(out_path))
    assert saved.vertices.shape == (n_tpl, 3)


@patch("autofacemonker._register.Facemarker")
@patch("autofacemonker._register.meshmonk")
def test_register_partial_landmarks(mock_meshmonk, mock_facemarker_cls, target_mesh):
    """Pipeline must not crash when only a subset of landmarks are detected."""
    mock_marker = MagicMock()
    mock_facemarker_cls.return_value = mock_marker

    monker = AutoFaceMonker()
    n_tpl = len(monker.template.vertices)

    # Only return 4 of the 7 default landmarks
    lmk_indices = [c[0] for c in DEFAULT_CORRESPONDENCES[:4]]
    tpl_vids = [c[1] for c in DEFAULT_CORRESPONDENCES[:4]]
    lmks_3d = {
        str(idx): monker.template.vertices[vid].tolist()
        for idx, vid in zip(lmk_indices, tpl_vids)
    }
    mock_pred = MagicMock()
    mock_pred.landmarks_3d = lmks_3d
    mock_marker.predict.return_value = mock_pred

    mock_nr = MagicMock()
    mock_nr.aligned_vertices = np.zeros((n_tpl, 3))
    mock_meshmonk.nonrigid_register.return_value = mock_nr

    result = monker.register(target_mesh)
    assert result.shape == (n_tpl, 3)


# ---------------------------------------------------------------------------
# Target-crop guard
# ---------------------------------------------------------------------------

def _target_with_distant_blob(monker, tmp_path):
    """Template geometry (perfectly covered) plus a far-away blob standing in
    for neck/ears/hair. Exactly `len(template.faces)` faces fall inside the
    crop shell, which is precisely the case the old `>` guard rejected."""
    tpl = trimesh.Trimesh(
        vertices=monker.template.vertices.copy(),
        faces=monker.template.faces.copy(),
        process=False,
    )
    blob = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
    blob.apply_translation([50.0, 0.0, 0.0])          # far outside any margin
    target = trimesh.util.concatenate([tpl, blob])
    path = tmp_path / "target_with_blob.obj"
    target.export(str(path))
    return str(path), len(tpl.faces), len(target.faces)


@patch("autofacemonker._register.Facemarker")
@patch("autofacemonker._register.meshmonk")
def test_crop_applies_when_shell_equals_template_face_count(
    mock_meshmonk, mock_facemarker_cls, tmp_path
):
    """Regression: the over-crop guard used to be

        if keep_f.sum() > len(self.template.faces):

    which compares a TARGET face count against the TEMPLATE's own 14050. A
    coarse scan puts few faces in the shell even when the shell covers the face
    perfectly, so the crop was skipped ENTIRELY and MeshMonk was handed a whole
    head. Measured on production this fired on 4 of 7 detail=medium scans.
    Here exactly len(template.faces) faces fall inside the shell -- `>` is
    false by one -- so the old code passed the blob straight through.
    """
    mock_marker = MagicMock()
    mock_facemarker_cls.return_value = mock_marker
    monker = AutoFaceMonker()
    n_tpl = len(monker.template.vertices)

    path, face_faces, total_faces = _target_with_distant_blob(monker, tmp_path)
    assert face_faces == len(monker.template.faces)      # the off-by-one case
    assert total_faces > face_faces                      # blob really is extra

    mock_marker.predict.return_value = _make_landmark_mock(monker)
    mock_nr = MagicMock()
    mock_nr.aligned_vertices = np.zeros((n_tpl, 3))
    mock_meshmonk.nonrigid_register.return_value = mock_nr

    monker.register(path)

    kwargs = mock_meshmonk.nonrigid_register.call_args.kwargs
    passed_faces = len(kwargs["target_faces"])
    assert passed_faces < total_faces, (
        "crop was skipped: MeshMonk received the full uncropped target"
    )
    assert passed_faces == face_faces


@patch("autofacemonker._register.Facemarker")
@patch("autofacemonker._register.meshmonk")
def test_crop_disabled_passes_full_target(mock_meshmonk, mock_facemarker_cls, tmp_path):
    """crop_margin=None must still mean 'no crop', unchanged."""
    mock_marker = MagicMock()
    mock_facemarker_cls.return_value = mock_marker
    monker = AutoFaceMonker(crop_margin=None)
    n_tpl = len(monker.template.vertices)

    path, _face_faces, total_faces = _target_with_distant_blob(monker, tmp_path)
    mock_marker.predict.return_value = _make_landmark_mock(monker)
    mock_nr = MagicMock()
    mock_nr.aligned_vertices = np.zeros((n_tpl, 3))
    mock_meshmonk.nonrigid_register.return_value = mock_nr

    monker.register(path)

    kwargs = mock_meshmonk.nonrigid_register.call_args.kwargs
    assert len(kwargs["target_faces"]) == total_faces
