"""BOS setup geometry: sensitivity, resolution, blur and pattern sizing.

Paraxial thin-lens model with the camera focused on the background (Raffel 2015,
Exp Fluids 56:60, Eqs. 1-8). Distances:

    camera ---- Z_A ---- object (flow) ---- Z_D ---- background
    |<-------------------- Z_B = Z_A + Z_D ------------------->|

All lengths are metres, angles radians, image quantities pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

FULL_FRAME_WIDTH_M = 36e-3


@dataclass(frozen=True)
class Camera:
    """A camera as seen by the processing: real focal length and effective pixel pitch.

    ``sensor_width_m`` is the width of the sensor area that maps onto ``image_width_px``
    (video modes often crop or bin, which changes the effective pitch).
    """

    focal_length_m: float
    sensor_width_m: float
    image_width_px: int
    image_height_px: int
    name: str = "Custom camera"

    @property
    def pixel_pitch_m(self) -> float:
        return self.sensor_width_m / self.image_width_px

    @property
    def horizontal_fov_rad(self) -> float:
        return 2.0 * math.atan(self.sensor_width_m / (2.0 * self.focal_length_m))

    @classmethod
    def from_equivalent_focal_length(
        cls,
        f_equiv_m: float,
        image_width_px: int,
        image_height_px: int,
        crop_factor: float = 1.0,
        name: str = "Custom camera",
    ) -> Camera:
        """Build from a 35 mm-equivalent focal length (what phones report)."""
        return cls(
            focal_length_m=f_equiv_m / crop_factor,
            sensor_width_m=FULL_FRAME_WIDTH_M / crop_factor,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            name=name,
        )

    def with_focal_length(self, focal_length_m: float) -> Camera:
        return replace(self, focal_length_m=focal_length_m)


@dataclass(frozen=True)
class Display:
    """A monitor or TV used to show the background pattern."""

    width_px: int
    height_px: int
    diagonal_m: float
    name: str = "Display"

    @property
    def pixel_pitch_m(self) -> float:
        diag_px = math.hypot(self.width_px, self.height_px)
        return self.diagonal_m / diag_px

    @property
    def width_m(self) -> float:
        return self.pixel_pitch_m * self.width_px

    @property
    def height_m(self) -> float:
        return self.pixel_pitch_m * self.height_px


@dataclass(frozen=True)
class Setup:
    camera: Camera
    z_a: float  # camera to object [m]
    z_d: float  # object to background [m]
    f_number: float = 8.0
    wavelength_m: float = 550e-9

    @property
    def z_b(self) -> float:
        return self.z_a + self.z_d

    @property
    def f(self) -> float:
        return self.camera.focal_length_m

    @property
    def p(self) -> float:
        return self.camera.pixel_pitch_m

    # --- magnification and scale -------------------------------------------------------
    @property
    def magnification_background(self) -> float:
        return self.f / (self.z_b - self.f)

    @property
    def magnification_object(self) -> float:
        """Chief-ray magnification of the (defocused) flow plane.

        The lens is focused on the background, so the sensor sits at b = f Z_B / (Z_B - f) and a
        point in the flow plane projects through the lens centre with magnification b / Z_A.
        (f / (Z_A - f) would only hold if the camera were focused on the flow.)
        """
        b = self.f * self.z_b / (self.z_b - self.f)
        return b / self.z_a

    @property
    def object_pixel_size_m(self) -> float:
        """Size of one image pixel in the object (flow) plane."""
        return self.p / self.magnification_object

    @property
    def background_pixel_size_m(self) -> float:
        """Size of one image pixel on the background."""
        return self.p / self.magnification_background

    @property
    def object_field_of_view_m(self) -> tuple[float, float]:
        h = self.object_pixel_size_m
        return h * self.camera.image_width_px, h * self.camera.image_height_px

    @property
    def background_field_of_view_m(self) -> tuple[float, float]:
        s = self.background_pixel_size_m
        return s * self.camera.image_width_px, s * self.camera.image_height_px

    # --- sensitivity ---------------------------------------------------------------------
    @property
    def pixels_per_radian(self) -> float:
        """Image displacement per unit deflection angle: f Z_D / ((Z_B - f) p)."""
        return self.f * self.z_d / ((self.z_b - self.f) * self.p)

    def deflection_from_pixels(self, displacement_px):
        return displacement_px / self.pixels_per_radian

    def pixels_from_deflection(self, deflection_rad):
        return deflection_rad * self.pixels_per_radian

    # --- blur and resolution -------------------------------------------------------------
    @property
    def aperture_diameter_m(self) -> float:
        return self.f / self.f_number

    @property
    def object_blur_m(self) -> float:
        """Footprint of the ray cone in the object plane: (f/N) Z_D / Z_B.

        This limits the spatial resolution of the measured flow.
        """
        return self.aperture_diameter_m * self.z_d / self.z_b

    @property
    def object_blur_px(self) -> float:
        return self.object_blur_m / self.object_pixel_size_m

    @property
    def diffraction_spot_px(self) -> float:
        """Airy-disk diameter of a background point on the sensor: 2.44 N (M+1) lambda."""
        m = self.magnification_background
        return 2.44 * self.f_number * (m + 1.0) * self.wavelength_m / self.p

    def background_feature_size_m(self, size_px: float) -> float:
        """Physical size on the background that images to ``size_px`` pixels."""
        return size_px * self.background_pixel_size_m

    def displacement_for_temperature_rise(
        self,
        delta_t_k: float,
        ambient_k: float = 293.15,
        k_gd: float = 2.26e-4,
        rho_ambient: float = 1.204,
    ) -> float:
        """Peak displacement [px] of an axisymmetric Gaussian hot plume.

        For delta_n(r) = A exp(-r^2/2s^2) the peak deflection is sqrt(2 pi) e^-1/2 A,
        independent of the plume width (derived in docs/research/physics-and-setup.md).
        """
        rho_hot = rho_ambient * ambient_k / (ambient_k + delta_t_k)
        amplitude = k_gd * (rho_hot - rho_ambient)
        eps = math.sqrt(2.0 * math.pi) * math.exp(-0.5) * abs(amplitude)
        return self.pixels_from_deflection(eps)


def diffraction_mtf(normalized_frequency: float) -> float:
    """MTF of an aberration-free circular aperture at frequency nu / nu_cutoff."""
    x = min(max(normalized_frequency, 0.0), 1.0)
    return (2.0 / math.pi) * (math.acos(x) - x * math.sqrt(1.0 - x * x))


class MoireRisk(Enum):
    NONE = "none"  # display grid unresolved (beyond optical cutoff) or well resolved
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class DisplayAnalysis:
    display_pixel_in_camera_px: float
    optical_cutoff_px: float
    moire_risk: MoireRisk
    block_size_display_px: int
    dot_size_camera_px: float
    pattern_field_coverage: tuple[float, float]  # fraction of camera frame covered (w, h)
    notes: tuple[str, ...]


def analyze_display(setup: Setup, display: Display, target_dot_px: float = 3.0) -> DisplayAnalysis:
    """Predict how a display looks to the camera and choose the pattern block size.

    The imaged display pixel pitch is M * p_disp. The grid is harmless when it is finer than
    the optical cutoff (diffraction spot ~ lambda N) or when each display pixel spans several
    camera pixels; in between it aliases into moire. Pattern elements are drawn as integer
    blocks of display pixels sized to ``target_dot_px`` on the sensor.
    """
    disp_px = setup.magnification_background * display.pixel_pitch_m / setup.p
    cutoff_px = setup.f_number * setup.wavelength_m / setup.p
    notes: list[str] = []

    grid_contrast = diffraction_mtf(cutoff_px / disp_px)
    if disp_px >= 4.0 or grid_contrast < 0.02:
        risk = MoireRisk.NONE
    elif disp_px >= 3.0 or grid_contrast < 0.1:
        risk = MoireRisk.LOW
    else:
        risk = MoireRisk.HIGH
    if risk is MoireRisk.HIGH:
        notes.append(
            "The display's pixel grid images to about "
            f"{disp_px:.2f} camera pixels, which aliases into moire. Move the camera, change "
            "the focal length, or defocus the background very slightly."
        )

    block = max(2, round(target_dot_px / disp_px)) if disp_px > 0 else 2
    dot_px = block * disp_px
    if dot_px < 2.0:
        notes.append(
            f"Pattern dots image to only {dot_px:.1f} px even at a {block}-pixel block. "
            "Move the background closer or use a longer focal length."
        )
    elif dot_px > 6.0:
        notes.append(
            f"The smallest usable dots image to {dot_px:.1f} px (blocks must be at least "
            "2 display pixels). Move the background further away, use a shorter focal length "
            "or a higher-resolution display."
        )

    fov_w, fov_h = setup.background_field_of_view_m
    coverage = (min(1.0, display.width_m / fov_w), min(1.0, display.height_m / fov_h))
    if min(coverage) < 1.0:
        notes.append(
            f"The display fills {coverage[0] * 100:.0f}% x {coverage[1] * 100:.0f}% of the "
            "frame. The flow must stay in front of the display area."
        )
    return DisplayAnalysis(disp_px, cutoff_px, risk, block, dot_px, coverage, tuple(notes))


def recommend_f_number(
    setup: Setup, resolution_m: float, max_dot_px: float = 4.0
) -> tuple[float, float]:
    """Aperture window: blur in the flow below ``resolution_m`` and diffraction below max dot.

    Returns (min_N, max_N). If min_N > max_N the requirements conflict.
    """
    min_n = setup.f * setup.z_d / (setup.z_b * resolution_m)
    m = setup.magnification_background
    max_n = max_dot_px * setup.p / (2.44 * setup.wavelength_m * (1.0 + m))
    return min_n, max_n


# Reference cameras used for realistic defaults (effective pitch in the given mode).
CAMERA_PRESETS: dict[str, Camera] = {
    "Sony A6400 · 24 MP still · 50 mm": Camera(
        50e-3, 23.5e-3, 6000, 4000, "Sony A6400 · 24 MP still"
    ),
    "Sony A6400 · 4K video · 50 mm": Camera(50e-3, 23.5e-3, 3840, 2160, "Sony A6400 · 4K video"),
    "Sony A6400 · 1080p video · 50 mm": Camera(
        50e-3, 23.5e-3, 1920, 1080, "Sony A6400 · 1080p video"
    ),
    "Pixel 8 Pro · main · 4K": Camera.from_equivalent_focal_length(
        24e-3, 3840, 2160, crop_factor=36 / 9.8, name="Pixel 8 Pro · main · 4K"
    ),
    "Pixel 8 Pro · 5x tele · 4K": Camera.from_equivalent_focal_length(
        113e-3, 3840, 2160, crop_factor=36 / 5.6, name="Pixel 8 Pro · 5x tele · 4K"
    ),
    "Generic phone · 4K": Camera.from_equivalent_focal_length(
        26e-3, 3840, 2160, crop_factor=6.0, name="Generic phone · 4K"
    ),
    "Generic APS-C · 4K · 50 mm": Camera(50e-3, 23.5e-3, 3840, 2160, "Generic APS-C · 4K"),
    "Generic full frame · 4K · 85 mm": Camera(85e-3, 36e-3, 3840, 2160, "Generic full frame · 4K"),
}

DISPLAY_PRESETS: dict[str, Display] = {
    '27" 1440p monitor': Display(2560, 1440, 27 * 0.0254, '27" 1440p'),
    '27" 4K monitor': Display(3840, 2160, 27 * 0.0254, '27" 4K'),
    '27" 3K monitor': Display(2880, 1620, 27 * 0.0254, '27" 3K'),
    '24" 1080p monitor': Display(1920, 1080, 24 * 0.0254, '24" 1080p'),
    '55" 4K TV': Display(3840, 2160, 55 * 0.0254, '55" 4K TV'),
    '65" 4K TV': Display(3840, 2160, 65 * 0.0254, '65" 4K TV'),
    '15.6" laptop 1080p': Display(1920, 1080, 15.6 * 0.0254, '15.6" laptop'),
}
