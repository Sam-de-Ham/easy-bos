"""Synthetic BOS capture: physically based rendering of background images seen through a field.

Forward model (convention used throughout easy-bos):

    I_flow(x) = I_ref(x + D(x)),   D = (f Z_D / ((Z_B - f) p)) * eps(x)   [px]

where eps = grad P is the exact deflection from ``fields.Field``. The background is a
continuous function (an 8x-oversampled pattern with bilinear interpolation, pre-blurred by
the optical PSF), so reference and distorted images sample *the same* texture and the true
displacement is exact. Each camera pixel integrates S x S sub-samples (fill factor 1).

Camera effects, all optional: depth-of-field blur of the displacement field (ray-cone footprint
in the object plane), display pixel grid (moire), photon shot noise, read noise, PRNU,
vignetting, quantisation, transfer curve, rolling shutter, PWM banding, camera shake and a real
video codec round trip through FFmpeg (PyAV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from easy_bos.bench.fields import Field
from easy_bos.core import imops
from easy_bos.core.device import default_device
from easy_bos.core.geometry import Camera, Setup
from easy_bos.core.patterns import PatternKind, PatternSpec, generate


@dataclass
class CameraModel:
    psf_sigma_px: float = 0.6
    full_well_e: float = 12000.0
    read_noise_e: float = 3.0
    prnu: float = 0.005
    exposure: float = 0.8  # white level as a fraction of full well
    black: float = 0.06  # display/print black level relative to white
    vignetting: float = 0.15  # relative fall-off at the frame corners
    bit_depth: int = 12
    transfer: str = "linear"  # "linear" or "bt709"
    subsamples: int = 3
    noise: bool = True

    @classmethod
    def ideal(cls) -> CameraModel:
        return cls(psf_sigma_px=0.5, noise=False, vignetting=0.0, bit_depth=16, black=0.0)


@dataclass
class CaptureEffects:
    shake_px: float = 0.0  # std of per-frame translation
    shake_rot_deg: float = 0.0
    shake_drift_px: float = 0.0  # linear drift over the sequence
    rolling_shutter: float = 0.0  # readout time as a fraction of the frame interval
    banding_amplitude: float = 0.0  # PWM row gain modulation
    banding_cycles: float = 7.3  # bands per frame height
    display_pitch_px: float = 0.0  # imaged display pixel pitch (0 = print, no grid)
    display_fill: float = 0.85  # display pixel aperture (fraction lit)
    codec: str | None = None  # None, "h264", "hevc10", "mjpeg"
    crf: int = 18


@dataclass
class Scene:
    field: Field
    setup: Setup
    pattern: PatternSpec = field(default_factory=lambda: PatternSpec(PatternKind.SPECKLE, 3))
    #: pattern element size in camera pixels (may be fractional)
    element_cam_px: float = 3.0
    apply_dof_blur: bool = True


@dataclass
class SyntheticCapture:
    frames: np.ndarray  # (T, H, W) float32 in [0, 1] as the camera would store it (decoded)
    reference_index: int  # index of a flow-free frame
    flow_frames: list[int]  # indices with the field present
    truth: dict[int, np.ndarray]  # per-frame true displacement (2, H, W) [px], DOF-blurred
    truth_sharp: dict[int, np.ndarray]  # without the depth-of-field blur
    rigid: dict[int, np.ndarray]  # per-frame 2x3 camera transform (pixel coords)
    times: np.ndarray
    scene: Scene
    camera: CameraModel
    effects: CaptureEffects


OVERSAMPLE = 8


class Renderer:
    """Renders frames of a scene on the GPU (or CPU)."""

    def __init__(self, scene: Scene, camera: CameraModel | None = None,
                 effects: CaptureEffects | None = None, seed: int = 0, device=None):
        self.scene = scene
        self.cam = camera or CameraModel()
        self.fx = effects or CaptureEffects()
        self.device = device or default_device()
        self.rng = np.random.default_rng(seed)
        c = scene.setup.camera
        self.h, self.w = c.image_height_px, c.image_width_px
        self.margin = 24
        self._background = self._build_background()
        self._prnu = self._build_prnu()

    # ---------------------------------------------------------------- background
    def _build_background(self) -> torch.Tensor:
        sc = self.scene
        bh = (self.h + 2 * self.margin) * OVERSAMPLE
        bw = (self.w + 2 * self.margin) * OVERSAMPLE
        e_os = max(1, round(sc.element_cam_px * OVERSAMPLE))
        tex = generate(sc.pattern.with_(element_px=e_os), bw, bh)
        bg = torch.from_numpy(tex).to(self.device)[None, None]
        if self.fx.display_pitch_px > 0:
            # Display pixel structure: lit aperture with dark gaps, period in camera px.
            period = self.fx.display_pitch_px * OVERSAMPLE
            coords = (torch.arange(bw, device=self.device) + 0.5) / period
            frac_x = coords - torch.floor(coords)
            coords_y = (torch.arange(bh, device=self.device) + 0.5) / period
            frac_y = coords_y - torch.floor(coords_y)
            fill = math.sqrt(self.fx.display_fill)
            mask = (frac_y[:, None] < fill).float() * (frac_x[None, :] < fill).float()
            bg = bg * mask[None, None] / self.fx.display_fill
        bg = self.cam.black + (1.0 - self.cam.black) * bg
        return imops.gaussian_blur(bg, self.cam.psf_sigma_px * OVERSAMPLE)

    def _build_prnu(self) -> torch.Tensor:
        if not self.cam.noise or self.cam.prnu <= 0:
            return torch.ones(1, 1, self.h, self.w, device=self.device)
        g = self.rng.normal(1.0, self.cam.prnu, (self.h, self.w)).astype(np.float32)
        return torch.from_numpy(g).to(self.device)[None, None]

    # ---------------------------------------------------------------- displacement
    def displacement(self, t: float, flow_on: bool) -> tuple[np.ndarray, np.ndarray]:
        """True displacement on a margin-padded pixel grid: (sharp, dof-blurred), (2, H+2m, W+2m)."""
        m = self.margin
        hh, ww = self.h + 2 * m, self.w + 2 * m
        if not flow_on:
            z = np.zeros((2, hh, ww), np.float32)
            return z, z
        st = self.scene.setup
        hpx = st.object_pixel_size_m
        ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float64)
        x_obj = (xs - m - (self.w - 1) / 2) * hpx
        y_obj = (ys - m - (self.h - 1) / 2) * hpx
        ex, ey = self.scene.field.deflection(x_obj, y_obj, t)
        d = np.stack([ex, ey]).astype(np.float64) * st.pixels_per_radian
        sharp = d.astype(np.float32)
        if self.scene.apply_dof_blur and st.object_blur_px > 1.0:
            dt = torch.from_numpy(sharp).to(self.device)[None]
            blurred = imops.disk_blur(dt, st.object_blur_px)[0].cpu().numpy()
        else:
            blurred = sharp
        return sharp, blurred

    # ---------------------------------------------------------------- frames
    def render(self, transform: np.ndarray, disp_slices: list[np.ndarray]) -> torch.Tensor:
        """Render one frame. ``disp_slices`` are padded displacement fields at rolling-shutter
        sample times (first = top row); ``transform`` maps frame pixels to scene pixels."""
        dev = self.device
        s = self.cam.subsamples
        m = self.margin
        offs = (torch.arange(s, device=dev, dtype=torch.float32) + 0.5) / s - 0.5
        ys = (torch.arange(self.h, device=dev, dtype=torch.float32)[:, None] + offs[None, :]).reshape(-1)
        xs = (torch.arange(self.w, device=dev, dtype=torch.float32)[:, None] + offs[None, :]).reshape(-1)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        a = torch.as_tensor(transform, dtype=torch.float32, device=dev)
        cx, cy = (self.w - 1) / 2, (self.h - 1) / 2
        xr, yr = xx - cx, yy - cy
        sx = a[0, 0] * xr + a[0, 1] * yr + a[0, 2] + cx
        sy = a[1, 0] * xr + a[1, 1] * yr + a[1, 2] + cy

        # Displacement at the scene position, blended across rolling-shutter time slices.
        if len(disp_slices) == 1:
            d = self._interp_disp(disp_slices[0], sx, sy)
        else:
            k = len(disp_slices)
            row_t = (yy / max(self.h - 1, 1)).clamp(0, 1) * (k - 1)
            d = torch.zeros(2, *sx.shape, device=dev)
            for i, sl in enumerate(disp_slices):
                wgt = torch.clamp(1.0 - torch.abs(row_t - i), min=0.0)
                if torch.any(wgt > 0):
                    d = d + wgt * self._interp_disp(sl, sx, sy)
        bx = sx + d[0]
        by = sy + d[1]
        u = (bx + m + 0.5) * OVERSAMPLE - 0.5
        v = (by + m + 0.5) * OVERSAMPLE - 0.5
        img = imops.sample(self._background, u[None], v[None], mode="bilinear")
        img = torch.nn.functional.avg_pool2d(img, s)  # pixel integration
        return img

    def _interp_disp(self, disp: np.ndarray, sx, sy) -> torch.Tensor:
        dt = torch.from_numpy(disp).to(self.device)[None]
        m = self.margin
        out = imops.sample(dt, (sx + m)[None], (sy + m)[None], mode="bicubic")
        return out[0]

    def expose(self, img: torch.Tensor, frame_index: int) -> torch.Tensor:
        """Sensor and pipeline: vignetting, banding, noise, PRNU, quantisation, transfer."""
        cam, fx = self.cam, self.fx
        dev = img.device
        h, w = img.shape[-2:]
        if cam.vignetting > 0:
            xx, yy = imops.pixel_grid(h, w, dev)
            r2 = ((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / (((w - 1) / 2) ** 2 + ((h - 1) / 2) ** 2)
            img = img * (1.0 - cam.vignetting * r2)
        if fx.banding_amplitude > 0:
            phase = self.rng.uniform(0, 2 * math.pi)
            rows = torch.arange(h, device=dev, dtype=torch.float32)
            gain = 1.0 + fx.banding_amplitude * torch.sin(2 * math.pi * fx.banding_cycles * rows / h + phase)
            img = img * gain[None, None, :, None]
        electrons = img * cam.exposure * cam.full_well_e * self._prnu
        if cam.noise:
            electrons = torch.poisson(torch.clamp(electrons, min=0.0))
            electrons = electrons + torch.randn_like(electrons) * cam.read_noise_e
        signal = torch.clamp(electrons / cam.full_well_e, 0.0, 1.0)
        if cam.transfer == "bt709":
            signal = torch.where(signal < 0.018, 4.5 * signal, 1.099 * signal.pow(0.45) - 0.099)
        levels = 2**cam.bit_depth - 1
        return torch.round(signal * levels) / levels

    def sequence(self, n_ref: int, n_flow: int, fps: float = 30.0) -> SyntheticCapture:
        """``n_ref`` flow-free frames followed by ``n_flow`` frames with the field present."""
        fx = self.fx
        n = n_ref + n_flow
        times = np.arange(n) / fps
        frames, truth, sharp_truth, rigid = [], {}, {}, {}
        drift = np.linspace(0.0, fx.shake_drift_px, n)
        for k in range(n):
            flow_on = k >= n_ref
            tx = self.rng.normal(0, fx.shake_px) + drift[k] if fx.shake_px or fx.shake_drift_px else 0.0
            ty = self.rng.normal(0, fx.shake_px) if fx.shake_px else 0.0
            th = math.radians(self.rng.normal(0, fx.shake_rot_deg)) if fx.shake_rot_deg else 0.0
            a = np.array([[math.cos(th), -math.sin(th), tx], [math.sin(th), math.cos(th), ty]])
            if fx.rolling_shutter > 0 and flow_on:
                slices_t = times[k] + np.linspace(0, fx.rolling_shutter / fps, 5)
            else:
                slices_t = np.array([times[k]])
            sharp_blur = [self.displacement(float(t), flow_on) for t in slices_t]
            img = self.render(a, [b for _, b in sharp_blur])
            frames.append(self.expose(img, k)[0, 0].cpu().numpy())
            rigid[k] = a
            if flow_on:
                mid = sharp_blur[len(sharp_blur) // 2]
                truth[k] = self._truth_in_frame(mid[1], a)
                sharp_truth[k] = self._truth_in_frame(mid[0], a)
        stack = np.stack(frames).astype(np.float32)
        if fx.codec:
            stack = codec_roundtrip(stack, fx.codec, fx.crf, fps)
        return SyntheticCapture(stack, 0, list(range(n_ref, n)), truth, sharp_truth, rigid,
                                times, self.scene, self.cam, fx)

    def _truth_in_frame(self, disp_padded: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Field displacement at the frame's scene positions: the residual left once the rigid
        camera motion (scene = A x) is removed. Vectors are in reference (scene) pixel axes."""
        dev = self.device
        xx, yy = imops.pixel_grid(self.h, self.w, dev)
        cx, cy = (self.w - 1) / 2, (self.h - 1) / 2
        at = torch.as_tensor(a, dtype=torch.float32, device=dev)
        sx = at[0, 0] * (xx - cx) + at[0, 1] * (yy - cy) + at[0, 2] + cx
        sy = at[1, 0] * (xx - cx) + at[1, 1] * (yy - cy) + at[1, 2] + cy
        return self._interp_disp(disp_padded, sx, sy).cpu().numpy()


def codec_roundtrip(frames: np.ndarray, codec: str, crf: int, fps: float) -> np.ndarray:
    """Encode and decode greyscale frames with a real FFmpeg encoder via PyAV."""
    import io

    import av

    buf = io.BytesIO()
    t, h, w = frames.shape
    he, we = h - h % 2, w - w % 2
    spec = {
        "h264": ("libx264", "yuv420p", 8, "mp4"),
        "hevc10": ("libx265", "yuv420p10le", 10, "mp4"),
        "mjpeg": ("mjpeg", "yuvj420p", 8, "avi"),
    }[codec]
    enc_name, pix_fmt, bits, container = spec
    with av.open(buf, mode="w", format=container) as out:
        stream = out.add_stream(enc_name, rate=round(fps))
        stream.width, stream.height, stream.pix_fmt = we, he, pix_fmt
        if enc_name == "libx264":
            stream.options = {"crf": str(crf), "preset": "medium"}
        elif enc_name == "libx265":
            stream.options = {"crf": str(crf), "preset": "medium", "x265-params": "log-level=error"}
        else:
            stream.options = {"q:v": str(max(2, crf // 6))}
        for k in range(t):
            y = frames[k, :he, :we]
            if bits == 8:
                plane = np.round(np.clip(y, 0, 1) * 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(plane, format="gray")
            else:
                plane = np.round(np.clip(y, 0, 1) * 1023).astype(np.uint16)
                frame = av.VideoFrame.from_ndarray(plane << 6, format="gray16le")
            frame = frame.reformat(format=pix_fmt)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    buf.seek(0)
    decoded = []
    with av.open(buf, mode="r") as inp:
        for fr in inp.decode(video=0):
            if bits == 8:
                decoded.append(fr.to_ndarray(format="gray").astype(np.float32) / 255.0)
            else:
                decoded.append((fr.to_ndarray(format="gray16le") >> 6).astype(np.float32) / 1023.0)
    out_frames = np.stack(decoded)
    if (he, we) != (h, w):
        out_frames = np.pad(out_frames, ((0, 0), (0, h - he), (0, w - we)), mode="edge")
    return out_frames


def bench_camera(width: int = 640, height: int = 480, focal_length_m: float = 50e-3,
                 pixel_pitch_m: float = 6.1e-6) -> Camera:
    """A small benchmark camera with the angular resolution of an APS-C 4K video frame."""
    return Camera(focal_length_m, pixel_pitch_m * width, width, height, "Benchmark camera")


def bench_setup(camera: Camera | None = None, z_a: float = 1.0, z_d: float = 1.0,
                f_number: float = 8.0) -> Setup:
    return Setup(camera or bench_camera(), z_a, z_d, f_number)
