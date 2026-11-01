"""A synthetic demo capture: a swaying hot plume in front of a speckle background.

Used by Help → Open demo, the screenshots and the end-to-end UI tests. The matching project
setup is returned alongside, so density and temperature come out in physical units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from easy_bos.bench import fields, synth
from easy_bos.core.export import write_video
from easy_bos.core.project import Project


def demo_project(width: int = 960, height: int = 540) -> Project:
    p = Project()
    s = p.setup
    s.camera_name = "Demo camera (synthetic)"
    s.focal_length_mm = 50.0
    s.sensor_width_mm = width * 12.2e-3  # 12.2 um effective pitch, like APS-C 1080p video
    s.image_width_px, s.image_height_px = width, height
    s.z_a_m, s.z_d_m, s.f_number = 1.0, 1.0, 11.0
    p.process.reference_range = (0, 10)
    return p


def write_demo(
    path: str | Path,
    width: int = 960,
    height: int = 540,
    n_ref: int = 10,
    n_flow: int = 50,
    fps: float = 25.0,
) -> Project:
    """Render the demo to ``path`` (MP4, H.264 at high quality) and return its project."""
    project = demo_project(width, height)
    setup = project.setup.setup()
    plume = fields.AxisymmetricPlume(
        delta_rho0=0.35,
        sigma0=0.006,
        spread=0.08,
        y_source=0.045,
        sway_amplitude=0.004,
        sway_period=1.2,
    )
    scene = synth.Scene(plume, setup, element_cam_px=3.0)
    cam = synth.CameraModel(full_well_e=9000, read_noise_e=2.5, bit_depth=10, transfer="bt709")
    cap = synth.Renderer(scene, cam, synth.CaptureEffects(shake_px=0.15), seed=7).sequence(
        n_ref, n_flow, fps
    )
    frames = (
        np.repeat((np.clip(f, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=2)
        for f in cap.frames
    )
    # High quality (as a careful capture would be); lower CRFs show codec blocking in BOS.
    write_video(path, frames, fps, crf=6)
    project.capture.paths = [str(path)]
    return project
