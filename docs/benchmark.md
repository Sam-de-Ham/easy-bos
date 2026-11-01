# Synthetic benchmark

Generated 2026-11-01 by `uv run easy-bos-bench` on NVIDIA GeForce RTX 4070 Laptop GPU · 8 GB · CUDA, 512×384 px frames. Every method runs through the production pipeline with its default parameters. Displacement errors are endpoint errors in pixels over the frame minus a 16-px border; *rel* is mean EPE divided by the mean true magnitude; *peak* is the ratio of the 99th-percentile magnitudes (1 = no attenuation). Temperature error is the RMS over the region more than 20 % of the peak rise above ambient. *Oracle* reconstructs from the true (depth-of-field blurred) displacement: its error is the floor set by optics and reconstruction, not by the estimator.

## Mean endpoint error (px), lower is better

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.0240 | 0.0054 | 0.0009 | 0.0020 | **0.0002** | 0.0002 | 0.0460 |
| Plume, RAW stills | 0.0248 | 0.0066 | 0.0033 | 0.0045 | **0.0031** | 0.0031 | 0.0464 |
| Plume, phone H.264 | **0.0397** | 0.0419 | 0.0434 | 0.0451 | 0.0444 | 0.0446 | 0.0606 |
| Plume, 10-bit HEVC | **0.0360** | 0.0379 | 0.0402 | 0.0414 | 0.0408 | 0.0410 | 0.0603 |
| Weak disturbance (breath), RAW | 0.0063 | 0.0033 | **0.0027** | 0.0034 | 0.0028 | 0.0028 | 0.0186 |
| Weak disturbance (breath), H.264 | 0.0130 | 0.0130 | **0.0129** | 0.0129 | 0.0129 | 0.0129 | 0.0156 |
| Turbulent multi-scale field | 0.0313 | 0.0126 | 0.0049 | 0.0075 | **0.0032** | 0.0032 | 0.1204 |
| Weak lens (calibration object) | 0.0392 | 0.0376 | 0.0265 | 0.0234 | **0.0178** | 0.0227 | 0.1648 |
| Heated vertical plate (planar) | 0.0394 | 0.0400 | 0.0164 | 0.0250 | **0.0080** | 0.0095 | 0.4301 |
| Camera shake | 0.0269 | 0.0068 | 0.0054 | 0.0049 | **0.0036** | 0.0048 | 0.0429 |
| Rolling shutter, unsteady plume | 0.0252 | 0.0070 | 0.0036 | 0.0048 | **0.0034** | 0.0034 | 0.0452 |
| Display flicker banding | 0.0250 | 0.0068 | 0.0033 | 0.0045 | 0.0032 | **0.0032** | 0.0469 |
| Display moiré | 0.0292 | 0.0077 | **0.0038** | 0.0050 | 0.0039 | 0.0039 | 0.0438 |
| Random-dot background | 0.0287 | 0.0077 | 0.0036 | 0.0045 | **0.0036** | 0.0036 | 0.0487 |
| Multi-scale noise background | 0.0187 | 0.0087 | **0.0053** | 0.0072 | 0.0054 | 0.0055 | 0.0403 |

## Relative error and peak recovery

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.186 / 1.13 | 0.042 / 1.01 | 0.007 / 0.99 | 0.016 / 1.00 | 0.001 / 1.00 | 0.001 / 1.00 | 0.357 / 0.79 |
| Plume, RAW stills | 0.192 / 1.13 | 0.051 / 1.01 | 0.025 / 1.00 | 0.035 / 1.00 | 0.024 / 1.00 | 0.024 / 1.00 | 0.360 / 0.77 |
| Plume, phone H.264 | 0.308 / 0.98 | 0.325 / 0.92 | 0.337 / 0.92 | 0.350 / 0.95 | 0.344 / 0.89 | 0.346 / 0.94 | 0.470 / 0.59 |
| Plume, 10-bit HEVC | 0.279 / 0.97 | 0.294 / 0.89 | 0.311 / 0.88 | 0.321 / 0.88 | 0.317 / 0.87 | 0.318 / 0.89 | 0.468 / 0.63 |
| Weak disturbance (breath), RAW | 0.486 / 1.40 | 0.253 / 1.11 | 0.211 / 1.03 | 0.264 / 1.05 | 0.217 / 1.04 | 0.219 / 1.04 | 1.443 / 0.97 |
| Weak disturbance (breath), H.264 | 1.011 / 0.12 | 1.005 / 0.07 | 1.004 / 0.06 | 1.004 / 0.08 | 1.004 / 0.07 | 1.004 / 0.07 | 1.210 / 0.59 |
| Turbulent multi-scale field | 0.158 / 0.97 | 0.064 / 0.97 | 0.025 / 0.99 | 0.038 / 0.99 | 0.016 / 1.00 | 0.016 / 1.00 | 0.609 / 0.74 |
| Weak lens (calibration object) | 0.118 / 0.84 | 0.113 / 0.90 | 0.080 / 0.93 | 0.071 / 0.95 | 0.054 / 0.97 | 0.068 / 0.97 | 0.496 / 0.61 |
| Heated vertical plate (planar) | 0.089 / 0.96 | 0.090 / 0.94 | 0.037 / 0.98 | 0.056 / 0.98 | 0.018 / 0.99 | 0.021 / 0.99 | 0.971 / 0.10 |
| Camera shake | 0.209 / 1.14 | 0.053 / 1.01 | 0.042 / 0.99 | 0.038 / 1.00 | 0.028 / 1.00 | 0.037 / 1.01 | 0.333 / 0.82 |
| Rolling shutter, unsteady plume | 0.196 / 1.13 | 0.054 / 1.00 | 0.028 / 0.99 | 0.037 / 1.00 | 0.027 / 1.00 | 0.027 / 1.00 | 0.350 / 0.79 |
| Display flicker banding | 0.194 / 1.13 | 0.053 / 1.01 | 0.026 / 1.00 | 0.035 / 1.00 | 0.025 / 1.00 | 0.025 / 1.00 | 0.364 / 0.77 |
| Display moiré | 0.232 / 1.15 | 0.061 / 1.02 | 0.030 / 1.00 | 0.039 / 1.00 | 0.031 / 1.00 | 0.031 / 1.01 | 0.347 / 0.86 |
| Random-dot background | 0.223 / 1.15 | 0.060 / 1.02 | 0.028 / 1.00 | 0.035 / 1.00 | 0.028 / 1.00 | 0.028 / 1.00 | 0.378 / 0.78 |
| Multi-scale noise background | 0.145 / 1.07 | 0.068 / 1.00 | 0.041 / 0.99 | 0.056 / 1.00 | 0.042 / 1.00 | 0.043 / 1.00 | 0.313 / 0.81 |

## Temperature RMS error (K) after reconstruction

| Scenario | Oracle | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.92 | 9.39 | 0.90 | 1.33 | 1.21 | 0.97 | 0.98 | 13.53 |
| Plume, RAW stills | 0.92 | 9.38 | 0.89 | 1.13 | 1.01 | 0.77 | 0.77 | 14.49 |
| Plume, phone H.264 | 0.92 | 5.42 | 10.55 | 11.81 | 11.60 | 11.65 | 11.50 | 24.85 |
| Plume, 10-bit HEVC | 0.92 | 3.98 | 8.00 | 9.11 | 8.90 | 8.86 | 8.81 | 22.45 |
| Heated vertical plate (planar) | 0.14 | 1.34 | 0.83 | 0.38 | 0.43 | 0.71 | 0.23 | 68.35 |
| Camera shake | 0.92 | 10.76 | 1.00 | 1.47 | 1.26 | 0.69 | 0.63 | 13.83 |

## Runtime per frame (s)

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.69 | 0.79 | 0.85 | 0.96 | 3.47 | 16.47 | 1.79 |
| Plume, RAW stills | 1.30 | 1.36 | 1.32 | 1.67 | 4.18 | 8.79 | 1.42 |
| Plume, phone H.264 | 0.53 | 0.62 | 0.58 | 0.93 | 3.38 | 6.57 | 0.65 |
| Plume, 10-bit HEVC | 0.57 | 0.70 | 0.60 | 0.96 | 3.34 | 12.79 | 0.71 |
| Weak disturbance (breath), RAW | 0.80 | 0.69 | 0.69 | 1.14 | 3.45 | 10.45 | 0.80 |
| Weak disturbance (breath), H.264 | 0.66 | 0.83 | 0.59 | 0.93 | 3.43 | 8.48 | 0.67 |
| Turbulent multi-scale field | 3.37 | 3.49 | 3.39 | 3.54 | 5.93 | 12.95 | 3.45 |
| Weak lens (calibration object) | 0.57 | 0.65 | 0.62 | 0.93 | 3.40 | 6.42 | 0.68 |
| Heated vertical plate (planar) | 1.22 | 1.30 | 1.34 | 1.54 | 4.06 | 18.36 | 1.34 |
| Camera shake | 2.33 | 2.47 | 2.39 | 2.66 | 5.21 | 10.21 | 2.40 |
| Rolling shutter, unsteady plume | 1.36 | 1.43 | 1.41 | 1.72 | 4.07 | 8.49 | 1.47 |
| Display flicker banding | 0.68 | 0.76 | 0.74 | 1.10 | 3.48 | 9.90 | 0.86 |
| Display moiré | 2.08 | 2.19 | 2.04 | 2.40 | 4.95 | 14.76 | 2.15 |
| Random-dot background | 1.27 | 1.36 | 1.35 | 1.75 | 5.71 | 22.56 | 1.57 |
| Multi-scale noise background | 1.35 | 1.62 | 1.38 | 2.30 | 4.56 | 10.32 | 1.44 |

## Scenarios

- **Plume, ideal camera** (`plume-ideal`): Axisymmetric plume (~+75 K), noise-free 16-bit sensor: algorithmic accuracy.
- **Plume, RAW stills** (`plume-raw`): 14-bit linear sensor with shot/read noise and PRNU, 4 averaged references.
- **Plume, phone H.264** (`plume-h264`): 8-bit BT.709 video, x264 CRF 20, phone-like noise and softer optics.
- **Plume, 10-bit HEVC** (`plume-hevc10`): 10-bit BT.709 video, x265 CRF 22 (mirrorless S-log/HLG-style capture).
- **Weak disturbance (breath), RAW** (`breath`): ~+8 K plume, peak displacement ~0.04 px, 14-bit stills, 16 averaged references: the noise floor.
- **Weak disturbance (breath), H.264** (`breath-h264`): The same ~+8 K plume through 8-bit H.264 at CRF 18: how much a consumer codec removes.
- **Turbulent multi-scale field** (`turbulent`): Random eddies 2-12 mm at f/16 and short Z_D (little blur): spatial resolution.
- **Weak lens (calibration object)** (`lens`): Thin lens F = 5 m placed 10 cm in front of the background: radial displacement linear in radius, up to ~2 px.
- **Heated vertical plate (planar)** (`plate`): 2-D natural-convection boundary layer, 15 cm deep: planar reconstruction.
- **Camera shake** (`shake`): Plume with 0.8 px rms shake and 0.05 deg rotation per frame; stabilisation on.
- **Rolling shutter, unsteady plume** (`rolling-shutter`): Plume swaying at 2 Hz (4 mm), sensor read out over 90 % of the frame interval at 30 fps; truth is the field at mid-readout.
- **Display flicker banding** (`banding`): 3 % PWM row banding seen through a rolling shutter; banding correction on.
- **Display moiré** (`moire`): Background on a display whose pixel grid images to 1.4 camera px at f/4.
- **Random-dot background** (`dots`): Same RAW plume with a sparse random-dot pattern instead of speckle.
- **Multi-scale noise background** (`multiscale`): Same RAW plume with a multi-scale noise pattern.
