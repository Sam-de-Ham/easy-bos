import numpy as np
import pytest

from easy_bos.bench import fields, synth
from easy_bos.core.reconstruct import abel
from easy_bos.core.reconstruct.density import FlowGeometry, ReconstructionSettings, reconstruct
from easy_bos.core.reconstruct.integrate import border_anchors, integrate_gradient


@pytest.mark.parametrize("n, tol_3pt, tol_direct", [(51, 3.5e-3, 1.0e-3), (101, 9e-4, 3e-4)])
def test_abel_operators_on_gaussian(n, tol_3pt, tol_direct):
    # Reference errors reproduced from the verified PyAbel/Dasch operators (docs/research).
    dr = 1.0 / (n - 1)
    r = np.arange(n) * dr
    s = 0.25
    f = np.exp(-(r**2) / s**2)
    p = np.sqrt(np.pi) * s * f  # exact Abel projection
    eps = -2 * r / s**2 * p
    assert np.abs(abel.invert_projection(p, dr) - f).max() < tol_3pt
    assert np.abs(abel.invert_deflection(eps) - f).max() < tol_direct
    assert np.abs(abel.forward_projection(f, dr) - p).max() < 5e-4


def test_axis_finder_subpixel():
    x = np.arange(200.0)
    c = 87.3
    row = -(x - c) * np.exp(-((x - c) ** 2) / (2 * 15**2))
    eps = np.tile(row, (10, 1))
    assert abel.find_axis(eps) == pytest.approx(c, abs=0.02)


def test_wls_integration_recovers_potential_with_masked_hole():
    h, w = 90, 120
    y, x = np.mgrid[0:h, 0:w] * 0.01
    f = np.exp(-((x - 0.6) ** 2 + (y - 0.45) ** 2) / 0.02)  # ~0 on the anchored border
    gx = -2 * (x - 0.6) / 0.02 * f
    gy = -2 * (y - 0.45) / 0.02 * f
    valid = np.ones((h, w), bool)
    valid[40:50, 20:30] = False
    sigma = np.ones((h, w))
    sigma[:, 60:] = 3.0  # noisier half
    res = integrate_gradient(gx, gy, 0.01, border_anchors((h, w), "lrtb"), sigma=sigma, valid=valid)
    err = np.nanmax(np.abs(res.phi - f))
    assert err < 3e-3
    assert np.isnan(res.phi[45, 25])


def _exact_displacement(field, setup):
    r = synth.Renderer(synth.Scene(field, setup, apply_dof_blur=False), synth.CameraModel.ideal())
    sharp, _ = r.displacement(0.0, True)
    m = r.margin
    d = sharp[:, m:-m, m:-m]
    h, w = d.shape[1:]
    hpx = setup.object_pixel_size_m
    ys, xs = np.mgrid[0:h, 0:w]
    return d, (xs - (w - 1) / 2) * hpx, (ys - (h - 1) / 2) * hpx


@pytest.mark.parametrize("method", ["indirect", "direct"])
def test_axisymmetric_temperature_from_exact_displacement(method):
    st = synth.bench_setup(synth.bench_camera(320, 240), f_number=16)
    fld = fields.AxisymmetricPlume(
        delta_rho0=0.4, sigma0=0.004, spread=0.05, x_axis=0.001, y_source=0.012
    )
    d, xo, yo = _exact_displacement(fld, st)
    rec = reconstruct(d, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC, abel_method=method))
    assert rec.axis_px == pytest.approx(
        (xo.shape[1] - 1) / 2 + 0.001 / st.object_pixel_size_m, abs=0.05
    )
    truth = fld.temperature(np.abs(xo - 0.001), yo)
    e = (rec.temperature - truth)[10:-10]
    assert np.sqrt(np.nanmean(e**2)) < 0.3  # K, on a ~120 K peak rise
    assert np.nanmax(np.abs(e)) < 1.5


def test_planar_temperature_from_exact_displacement():
    st = synth.bench_setup(synth.bench_camera(320, 240), f_number=16)
    fld = fields.PlanarBlob(sigma_x=0.004, sigma_y=0.006)
    d, xo, yo = _exact_displacement(fld, st)
    rec = reconstruct(
        d, st, ReconstructionSettings(FlowGeometry.PLANAR, depth_m=fld.depth, anchor_sides="lr")
    )
    e = rec.temperature - fld.temperature(xo, yo)
    assert np.sqrt(np.nanmean(e**2)) < 0.1
    assert np.nanmax(np.abs(e)) < 0.5


def _plume_case(shape=(320, 240), x_axis=0.001):
    st = synth.bench_setup(synth.bench_camera(*shape), f_number=16)
    fld = fields.AxisymmetricPlume(
        delta_rho0=0.4, sigma0=0.004, spread=0.05, x_axis=x_axis, y_source=0.012
    )
    d, xo, yo = _exact_displacement(fld, st)
    return st, fld, d, xo, yo


def test_masked_and_nan_input_is_handled():
    st, fld, d, xo, yo = _plume_case()
    d = d.copy()
    d[:, 100:110, 150:170] = np.nan  # hole through the plume
    valid = np.ones(d.shape[1:], bool)
    valid[:, :5] = False
    rec = reconstruct(d, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC), valid=valid)
    truth = fld.temperature(np.abs(xo - 0.001), yo)
    assert np.isnan(rec.temperature[105, 160]) and np.isnan(rec.temperature[50, 2])
    e = (rec.temperature - truth)[10:-10]
    assert np.sqrt(np.nanmean(e**2)) < 0.6
    assert rec.axis_px == pytest.approx((xo.shape[1] - 1) / 2 + 0.001 / st.object_pixel_size_m,
                                        abs=0.1)


@pytest.mark.parametrize("k", [2, 3])
def test_downsampled_axis_and_resolution(k):
    st, fld, d, xo, yo = _plume_case()
    true_axis = (xo.shape[1] - 1) / 2 + 0.001 / st.object_pixel_size_m
    rec = reconstruct(d, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC, downsample=k))
    assert rec.temperature.shape == d.shape[1:]
    assert rec.axis_px == pytest.approx(true_axis, abs=0.15)
    assert rec.pixel_size_m == pytest.approx(st.object_pixel_size_m)
    given = reconstruct(d, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC, downsample=k,
                                                      axis=true_axis))
    truth = fld.temperature(np.abs(xo - 0.001), yo)
    e = (given.temperature - truth)[10:-10]
    assert np.sqrt(np.nanmean(e**2)) < 0.6


def test_horizontal_axis_matches_vertical():
    st, fld, d, xo, yo = _plume_case((241, 321))
    rec_v = reconstruct(d, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC))
    dt = np.stack([d[1].T, d[0].T])  # the same flow rotated: axis now horizontal
    rec_h = reconstruct(dt, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC,
                                                       axis_orientation="horizontal",
                                                       anchor_sides="tb"))
    np.testing.assert_allclose(rec_h.temperature.T, rec_v.temperature, atol=1e-6,
                               equal_nan=True)


def test_anchor_on_flow_warns():
    st, fld, d, xo, yo = _plume_case()
    rec = reconstruct(d, st, ReconstructionSettings(FlowGeometry.AXISYMMETRIC, anchor_sides="t"))
    assert any("anchor" in w for w in rec.warnings)


def test_integration_converges_with_nonuniform_uncertainty():
    h, w = 240, 320
    y, x = np.mgrid[0:h, 0:w] / w
    f = np.exp(-((x - 0.5) ** 2 + (y - 0.35) ** 2) / 0.01)
    gx = -2 * (x - 0.5) / 0.01 * f
    gy = -2 * (y - 0.35) / 0.01 * f
    sig = np.ones((h, w))
    sig[:, w // 2 :] = 3.0
    res = integrate_gradient(gx, gy, 1 / w, border_anchors((h, w), "lrtb"), sigma=sig)
    assert res.solve.converged
    assert np.abs(res.phi - f).max() < 1e-3
