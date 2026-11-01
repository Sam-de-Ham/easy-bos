import math

import pytest

from easy_bos.core import physics
from easy_bos.core.geometry import (
    CAMERA_PRESETS,
    DISPLAY_PRESETS,
    Camera,
    MoireRisk,
    Setup,
    analyze_display,
    diffraction_mtf,
    recommend_f_number,
)


def test_gladstone_dale_matches_ciddor_values():
    assert physics.gladstone_dale_air(550e-9) == pytest.approx(2.268e-4, rel=2e-3)
    assert physics.gladstone_dale_air(633e-9) == pytest.approx(2.257e-4, rel=2e-3)


def test_air_density_and_temperature_roundtrip():
    rho = physics.air_density(101325.0, 293.15)
    assert rho == pytest.approx(1.204, rel=1e-3)
    t = physics.temperature_from_density(rho * 0.8, rho, 293.15)
    assert t == pytest.approx(293.15 / 0.8)


def test_sensitivity_formula():
    cam = Camera(50e-3, 6.0e-3, 1000, 800)  # 6 um pixels
    st = Setup(cam, z_a=1.0, z_d=1.0)
    # Raffel 2015 Eq. 3: dy = f Z_D / (Z_D + Z_A - f) * eps
    expected = 50e-3 * 1.0 / (2.0 - 50e-3) / 6e-6
    assert st.pixels_per_radian == pytest.approx(expected)
    assert st.deflection_from_pixels(st.pixels_from_deflection(1e-4)) == pytest.approx(1e-4)
    assert st.object_pixel_size_m == pytest.approx(6e-6 * (1.0 - 0.05) / 0.05)


def test_gaussian_plume_peak_displacement():
    cam = Camera(50e-3, 4e-6 * 6000, 6000, 4000)
    st = Setup(cam, 1.0, 1.0)
    # docs/research/physics-and-setup.md: ~0.67 px for a 100 K plume at 1 m / 1 m, 4 um pixels.
    assert st.displacement_for_temperature_rise(100.0) == pytest.approx(0.67, abs=0.03)


def test_blur_and_aperture_window():
    st = Setup(Camera(50e-3, 4e-6 * 1000, 1000, 1000), 1.0, 1.0, f_number=8)
    assert st.object_blur_m == pytest.approx(50e-3 / 8 * 0.5)
    lo, hi = recommend_f_number(st, resolution_m=4e-3)
    assert lo == pytest.approx(50e-3 * 1.0 / (2.0 * 4e-3))
    assert hi == pytest.approx(4.0 * 4e-6 / (2.44 * 550e-9 * (1 + st.magnification_background)))
    assert hi > lo
    # Asking for 2 mm resolution with these optics is contradictory: the window closes.
    lo2, hi2 = recommend_f_number(st, resolution_m=2e-3)
    assert lo2 > hi2


def test_mtf_limits():
    assert diffraction_mtf(0.0) == pytest.approx(1.0)
    assert diffraction_mtf(1.0) == pytest.approx(0.0, abs=1e-12)
    assert 0 < diffraction_mtf(0.5) < 1


def test_display_analysis_flags_aliasing_band():
    cam = CAMERA_PRESETS["Sony A6400 · 4K video · 50 mm"]
    tv = DISPLAY_PRESETS['55" 4K TV']
    target = 1.5  # one TV pixel -> 1.5 camera pixels
    z_b = cam.focal_length_m * (1 + tv.pixel_pitch_m / (target * cam.pixel_pitch_m))
    st = Setup(cam, z_a=z_b / 2, z_d=z_b / 2, f_number=4)
    a = analyze_display(st, tv)
    assert a.display_pixel_in_camera_px == pytest.approx(target, rel=1e-6)
    assert a.moire_risk is MoireRisk.HIGH
    assert a.block_size_display_px >= 2
    far = Setup(cam, z_a=10.0, z_d=10.0, f_number=16)
    assert analyze_display(far, tv).moire_risk is MoireRisk.NONE


def test_equivalent_focal_length_preserves_field_of_view():
    c = Camera.from_equivalent_focal_length(24e-3, 3840, 2160, crop_factor=36 / 9.8)
    assert c.horizontal_fov_rad == pytest.approx(2 * math.atan(18 / 24))
