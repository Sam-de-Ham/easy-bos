import h5py
import numpy as np
import pytest

from easy_bos.core import export, figures
from easy_bos.core.geometry import Camera, Setup
from easy_bos.core.media import open_source
from easy_bos.core.project import Project


def _rgb(h=64, w=96, k=0):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (np.arange(w) * 2 + k) % 256
    img[..., 1] = 128
    return img


@pytest.mark.parametrize("ext", ["mp4", "mov", "mkv"])
def test_video_export_formats(tmp_path, ext):
    path = tmp_path / f"v.{ext}"
    export.write_video(path, (_rgb(k=i) for i in range(5)), fps=25)
    src = open_source(path, channel="red")
    assert len(src) == 5
    assert src.info.width == 96 and src.info.height == 64


def test_png_and_tiff(tmp_path):
    export.write_png(tmp_path / "a.png", _rgb())
    data = np.linspace(-1, 1, 64 * 96, dtype=np.float32).reshape(64, 96)
    export.write_tiff_float(tmp_path / "a.tif", data)
    back = open_source(tmp_path / "a.tif").frame(0)
    np.testing.assert_allclose(back, data, atol=1e-6)


def test_field_hdf5_has_units_and_geometry(tmp_path):
    st = Setup(Camera(50e-3, 23.5e-3, 96, 64), 1.0, 1.0)
    d = np.random.default_rng(0).normal(0, 0.1, (2, 64, 96)).astype(np.float32)
    export.write_field_hdf5(tmp_path / "f.h5", d, st, density=np.ones((64, 96)))
    with h5py.File(tmp_path / "f.h5") as f:
        np.testing.assert_allclose(f["deflection_rad"][...], d / st.pixels_per_radian, rtol=1e-6)
        assert f.attrs["object_pixel_size_m"] == pytest.approx(st.object_pixel_size_m)
        assert "density_kg_m3" in f


def test_profile_csv(tmp_path):
    export.write_profile_csv(tmp_path / "p.csv", np.arange(3.0), {"T (K)": np.array([1, 2, 3.0])})
    lines = (tmp_path / "p.csv").read_text().splitlines()
    assert lines[0] == "distance,T (K)" and len(lines) == 4


def test_figure_and_print_pdf(tmp_path):
    figures.save_field_figure(
        tmp_path / "fig.png", _rgb(), 0.2, "Knife edge", "D (px)", (-0.5, 0.5), "gray"
    )
    assert (tmp_path / "fig.png").stat().st_size > 1000
    pattern = (np.random.default_rng(1).random((300, 400)) > 0.5).astype(np.float32)
    export.write_print_pattern_pdf(tmp_path / "p.pdf", pattern, 300, "test")
    assert (tmp_path / "p.pdf").read_bytes()[:4] == b"%PDF"


def test_project_roundtrip(tmp_path):
    p = Project()
    p.setup.z_d_m = 0.42
    p.process.roi = (1, 2, 30, 40)
    p.process.exclusions = [(5, 5, 10, 10)]
    p.reconstruct.geometry = "planar"
    p.save(tmp_path / "x.ebos")
    q = Project.load(tmp_path / "x.ebos")
    assert q.setup.z_d_m == 0.42 and q.process.roi == (1, 2, 30, 40)
    assert q.process.exclusions == [(5, 5, 10, 10)]
    s = q.process.settings((100, 120))
    assert s.mask is not None and s.mask.shape == (40, 30)
    assert s.mask[5, 5] == 0 and s.mask[0, 0] == 1
