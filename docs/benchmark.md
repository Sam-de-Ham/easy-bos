# Synthetic benchmark

Generated 2026-11-01 by `uv run easy-bos-bench` on NVIDIA GeForce RTX 4070 Laptop GPU · 8 GB · CUDA, 512×384 px frames. Every method runs through the production pipeline with its default parameters. Displacement errors are endpoint errors in pixels over the frame minus a 16-px border; *rel* is mean EPE divided by the mean true magnitude; *peak* is the ratio of the 99th-percentile magnitudes (1 = no attenuation). Temperature error is the RMS over the region more than 20 % of the peak rise above ambient. *Oracle* reconstructs from the true (depth-of-field blurred) displacement: its error is the floor set by optics and reconstruction, not by the estimator.

## Mean endpoint error (px), lower is better

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.0240 | 0.0054 | 0.0009 | 0.0020 | **0.0002** | 0.0002 | 0.0460 |
| Plume, RAW stills | 0.0249 | 0.0068 | 0.0032 | 0.0044 | 0.0030 | **0.0030** | 0.0469 |
| Plume, phone H.264 | **0.0393** | 0.0418 | 0.0434 | 0.0451 | 0.0443 | 0.0446 | 0.0615 |
| Plume, 10-bit HEVC | **0.0357** | 0.0377 | 0.0401 | 0.0412 | 0.0407 | 0.0409 | 0.0625 |
| Weak disturbance (breath), RAW | 0.0065 | 0.0032 | **0.0028** | 0.0034 | 0.0028 | 0.0028 | 0.0186 |
| Weak disturbance (breath), H.264 | 0.0131 | 0.0130 | **0.0130** | 0.0130 | 0.0130 | 0.0130 | 0.0158 |
| Turbulent multi-scale field | 0.0314 | 0.0126 | 0.0048 | 0.0075 | **0.0031** | 0.0032 | 0.1211 |
| Weak lens (calibration object) | 0.0393 | 0.0376 | 0.0265 | 0.0235 | **0.0182** | 0.0228 | 0.1593 |
| Heated vertical plate (planar) | 0.0391 | 0.0400 | 0.0165 | 0.0251 | **0.0075** | 0.0095 | 0.4272 |
| Camera shake | 0.0269 | 0.0067 | 0.0053 | 0.0050 | **0.0036** | 0.0047 | 0.0430 |
| Rolling shutter, unsteady plume | 0.0249 | 0.0069 | 0.0036 | 0.0047 | 0.0034 | **0.0034** | 0.0457 |
| Display flicker banding | 0.0252 | 0.0067 | 0.0032 | 0.0045 | 0.0032 | **0.0031** | 0.0454 |
| Display moiré | 0.0293 | 0.0075 | **0.0037** | 0.0049 | 0.0038 | 0.0038 | 0.0429 |
| Random-dot background | 0.0292 | 0.0080 | 0.0036 | 0.0045 | **0.0036** | 0.0036 | 0.0482 |
| Multi-scale noise background | 0.0187 | 0.0086 | **0.0053** | 0.0073 | 0.0055 | 0.0056 | 0.0403 |

## Relative error and peak recovery

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.186 / 1.13 | 0.042 / 1.01 | 0.007 / 0.99 | 0.016 / 1.00 | 0.001 / 1.00 | 0.001 / 1.00 | 0.357 / 0.79 |
| Plume, RAW stills | 0.193 / 1.13 | 0.053 / 1.01 | 0.024 / 1.00 | 0.034 / 1.00 | 0.023 / 1.00 | 0.023 / 1.00 | 0.364 / 0.77 |
| Plume, phone H.264 | 0.305 / 0.98 | 0.324 / 0.94 | 0.337 / 0.93 | 0.350 / 0.95 | 0.344 / 0.90 | 0.346 / 0.95 | 0.477 / 0.56 |
| Plume, 10-bit HEVC | 0.277 / 0.97 | 0.292 / 0.89 | 0.311 / 0.88 | 0.320 / 0.88 | 0.316 / 0.86 | 0.317 / 0.88 | 0.485 / 0.62 |
| Weak disturbance (breath), RAW | 0.503 / 1.43 | 0.250 / 1.12 | 0.215 / 1.03 | 0.266 / 1.06 | 0.219 / 1.05 | 0.219 / 1.05 | 1.442 / 0.93 |
| Weak disturbance (breath), H.264 | 1.017 / 0.13 | 1.006 / 0.07 | 1.005 / 0.06 | 1.006 / 0.08 | 1.007 / 0.08 | 1.006 / 0.07 | 1.223 / 0.63 |
| Turbulent multi-scale field | 0.159 / 0.97 | 0.064 / 0.97 | 0.024 / 0.99 | 0.038 / 0.99 | 0.016 / 1.00 | 0.016 / 1.00 | 0.612 / 0.73 |
| Weak lens (calibration object) | 0.118 / 0.84 | 0.113 / 0.90 | 0.080 / 0.93 | 0.071 / 0.95 | 0.055 / 0.97 | 0.069 / 0.97 | 0.480 / 0.63 |
| Heated vertical plate (planar) | 0.088 / 0.96 | 0.090 / 0.94 | 0.037 / 0.98 | 0.057 / 0.98 | 0.017 / 0.99 | 0.021 / 0.99 | 0.965 / 0.10 |
| Camera shake | 0.209 / 1.15 | 0.052 / 1.01 | 0.041 / 0.99 | 0.039 / 1.00 | 0.028 / 1.00 | 0.037 / 1.00 | 0.333 / 0.80 |
| Rolling shutter, unsteady plume | 0.193 / 1.13 | 0.054 / 1.01 | 0.028 / 0.99 | 0.037 / 1.00 | 0.026 / 1.00 | 0.026 / 1.00 | 0.354 / 0.80 |
| Display flicker banding | 0.195 / 1.13 | 0.052 / 1.01 | 0.025 / 1.00 | 0.035 / 1.00 | 0.025 / 1.00 | 0.024 / 1.00 | 0.353 / 0.78 |
| Display moiré | 0.232 / 1.15 | 0.059 / 1.02 | 0.029 / 1.00 | 0.039 / 1.00 | 0.031 / 1.00 | 0.030 / 1.00 | 0.341 / 0.88 |
| Random-dot background | 0.226 / 1.15 | 0.062 / 1.02 | 0.028 / 1.00 | 0.035 / 1.00 | 0.028 / 1.01 | 0.028 / 1.01 | 0.374 / 0.82 |
| Multi-scale noise background | 0.145 / 1.08 | 0.067 / 1.01 | 0.041 / 1.00 | 0.056 / 1.01 | 0.043 / 1.01 | 0.043 / 1.01 | 0.312 / 0.84 |

## Temperature RMS error (K) after reconstruction

| Scenario | Oracle | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.92 | 9.39 | 0.90 | 1.33 | 1.21 | 0.98 | 0.98 | 13.53 |
| Plume, RAW stills | 0.92 | 9.39 | 0.97 | 1.09 | 0.96 | 0.73 | 0.73 | 14.53 |
| Plume, phone H.264 | 0.92 | 6.35 | 10.97 | 12.15 | 11.99 | 12.10 | 11.95 | 24.95 |
| Plume, 10-bit HEVC | 0.92 | 4.33 | 8.60 | 9.70 | 9.53 | 9.51 | 9.46 | 23.60 |
| Heated vertical plate (planar) | 0.14 | 1.31 | 0.83 | 0.38 | 0.43 | 0.54 | 0.23 | 67.78 |
| Camera shake | 0.92 | 10.85 | 1.02 | 1.47 | 1.26 | 0.69 | 0.81 | 13.75 |

## Runtime per frame (s)

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.06 | 0.13 | 0.30 | 0.66 | 0.37 | 16.62 | 1.02 |
| Plume, RAW stills | 0.04 | 0.19 | 0.05 | 0.33 | 0.34 | 8.86 | 0.12 |
| Plume, phone H.264 | 0.04 | 0.13 | 0.06 | 0.49 | 0.46 | 6.12 | 0.14 |
| Plume, 10-bit HEVC | 0.04 | 0.11 | 0.04 | 0.28 | 0.28 | 5.84 | 0.20 |
| Weak disturbance (breath), RAW | 0.03 | 0.11 | 0.05 | 0.36 | 0.40 | 9.76 | 0.37 |
| Weak disturbance (breath), H.264 | 0.05 | 0.15 | 0.07 | 0.35 | 0.28 | 9.48 | 0.23 |
| Turbulent multi-scale field | 0.05 | 0.15 | 0.06 | 0.51 | 0.43 | 13.20 | 0.22 |
| Weak lens (calibration object) | 1.21 | 1.07 | 1.26 | 1.27 | 1.42 | 6.25 | 0.71 |
| Heated vertical plate (planar) | 0.66 | 0.63 | 0.58 | 0.84 | 1.24 | 12.28 | 0.81 |
| Camera shake | 1.68 | 1.54 | 1.23 | 1.83 | 1.66 | 6.13 | 1.34 |
| Rolling shutter, unsteady plume | 0.04 | 0.12 | 0.05 | 0.25 | 0.26 | 11.03 | 0.22 |
| Display flicker banding | 0.75 | 0.84 | 0.84 | 0.99 | 0.98 | 10.68 | 0.88 |
| Display moiré | 0.03 | 0.11 | 0.05 | 0.26 | 0.35 | 11.90 | 0.15 |
| Random-dot background | 0.04 | 0.12 | 0.04 | 0.26 | 0.24 | 10.52 | 0.12 |
| Multi-scale noise background | 0.04 | 0.11 | 0.04 | 0.25 | 0.27 | 5.68 | 0.22 |

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
