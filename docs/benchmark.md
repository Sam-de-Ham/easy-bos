# Synthetic benchmark

Generated 2026-11-01 by `uv run easy-bos-bench` on NVIDIA GeForce RTX 4070 Laptop GPU · 8 GB · CUDA, 512×384 px frames. Every method runs through the production pipeline with its default parameters. Displacement errors are endpoint errors in pixels over the frame minus a 16-px border; *rel* is mean EPE divided by the mean true magnitude; *peak* is the ratio of the 99th-percentile magnitudes (1 = no attenuation). Temperature error is the RMS over the region more than 20 % of the peak rise above ambient. *Oracle* reconstructs from the true (depth-of-field blurred) displacement: its error is the floor set by optics and reconstruction, not by the estimator.

## Mean endpoint error (px), lower is better

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.0240 | 0.0054 | 0.0009 | 0.0020 | **0.0002** | 0.0002 | 0.0460 |
| Plume, RAW stills | 0.0248 | 0.0067 | 0.0032 | 0.0045 | 0.0030 | **0.0030** | 0.0457 |
| Plume, phone H.264 | **0.0402** | 0.0420 | 0.0437 | 0.0453 | 0.0448 | 0.0448 | 0.0600 |
| Plume, 10-bit HEVC | **0.0360** | 0.0383 | 0.0406 | 0.0418 | 0.0412 | 0.0414 | 0.0615 |
| Weak disturbance (breath), RAW | 0.0063 | 0.0033 | **0.0027** | 0.0034 | 0.0028 | 0.0028 | 0.0186 |
| Weak disturbance (breath), H.264 | 0.0131 | 0.0130 | **0.0130** | 0.0130 | 0.0130 | 0.0130 | 0.0157 |
| Turbulent multi-scale field | 0.0315 | 0.0127 | 0.0050 | 0.0076 | **0.0031** | 0.0032 | 0.1205 |
| Weak lens (calibration object) | 0.0393 | 0.0376 | 0.0265 | 0.0235 | **0.0181** | 0.0227 | 0.1622 |
| Heated vertical plate (planar) | 0.0373 | 0.0382 | 0.0159 | 0.0242 | 0.0095 | **0.0095** | 0.4013 |
| Camera shake | 0.0272 | 0.0069 | 0.0052 | 0.0051 | **0.0036** | 0.0046 | 0.0425 |
| Rolling shutter, unsteady plume | 0.0253 | 0.0070 | 0.0035 | 0.0047 | **0.0033** | 0.0033 | 0.0451 |
| Display flicker banding | 0.0256 | 0.0069 | 0.0033 | 0.0046 | 0.0034 | **0.0033** | 0.0466 |
| Display moiré | 0.0291 | 0.0075 | **0.0037** | 0.0049 | 0.0039 | 0.0039 | 0.0438 |
| Random-dot background | 0.0288 | 0.0078 | 0.0036 | 0.0045 | 0.0036 | **0.0036** | 0.0489 |
| Multi-scale noise background | 0.0190 | 0.0086 | **0.0054** | 0.0073 | 0.0055 | 0.0055 | 0.0395 |

## Relative error and peak recovery

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.186 / 1.13 | 0.042 / 1.01 | 0.007 / 0.99 | 0.016 / 1.00 | 0.001 / 1.00 | 0.001 / 1.00 | 0.357 / 0.79 |
| Plume, RAW stills | 0.193 / 1.12 | 0.052 / 1.01 | 0.025 / 1.00 | 0.035 / 1.00 | 0.023 / 1.00 | 0.023 / 1.00 | 0.354 / 0.79 |
| Plume, phone H.264 | 0.312 / 1.01 | 0.326 / 0.96 | 0.339 / 0.94 | 0.352 / 0.95 | 0.347 / 0.92 | 0.348 / 0.96 | 0.465 / 0.61 |
| Plume, 10-bit HEVC | 0.280 / 0.99 | 0.297 / 0.89 | 0.315 / 0.89 | 0.324 / 0.89 | 0.319 / 0.89 | 0.321 / 0.90 | 0.477 / 0.62 |
| Weak disturbance (breath), RAW | 0.485 / 1.41 | 0.252 / 1.13 | 0.210 / 1.04 | 0.262 / 1.06 | 0.217 / 1.05 | 0.216 / 1.05 | 1.441 / 1.00 |
| Weak disturbance (breath), H.264 | 1.018 / 0.14 | 1.008 / 0.08 | 1.006 / 0.07 | 1.008 / 0.09 | 1.007 / 0.07 | 1.008 / 0.08 | 1.217 / 0.63 |
| Turbulent multi-scale field | 0.159 / 0.96 | 0.064 / 0.97 | 0.025 / 0.99 | 0.038 / 0.99 | 0.016 / 1.00 | 0.016 / 1.00 | 0.609 / 0.73 |
| Weak lens (calibration object) | 0.118 / 0.84 | 0.113 / 0.90 | 0.080 / 0.93 | 0.071 / 0.95 | 0.054 / 0.97 | 0.068 / 0.97 | 0.488 / 0.62 |
| Heated vertical plate (planar) | 0.084 / 0.93 | 0.086 / 0.92 | 0.036 / 0.96 | 0.055 / 0.96 | 0.022 / 0.95 | 0.021 / 0.97 | 0.906 / 0.10 |
| Camera shake | 0.211 / 1.15 | 0.054 / 1.01 | 0.041 / 0.99 | 0.039 / 1.00 | 0.028 / 1.00 | 0.036 / 1.00 | 0.329 / 0.80 |
| Rolling shutter, unsteady plume | 0.196 / 1.14 | 0.054 / 1.01 | 0.027 / 0.99 | 0.036 / 1.00 | 0.026 / 1.00 | 0.026 / 1.00 | 0.350 / 0.79 |
| Display flicker banding | 0.199 / 1.13 | 0.054 / 1.01 | 0.025 / 1.00 | 0.035 / 1.00 | 0.026 / 1.00 | 0.025 / 1.00 | 0.362 / 0.77 |
| Display moiré | 0.231 / 1.15 | 0.060 / 1.02 | 0.029 / 1.00 | 0.039 / 1.00 | 0.031 / 1.01 | 0.031 / 1.01 | 0.347 / 0.89 |
| Random-dot background | 0.223 / 1.15 | 0.061 / 1.02 | 0.028 / 1.00 | 0.035 / 1.00 | 0.028 / 1.01 | 0.028 / 1.01 | 0.379 / 0.78 |
| Multi-scale noise background | 0.148 / 1.08 | 0.067 / 1.00 | 0.042 / 0.99 | 0.056 / 1.00 | 0.042 / 1.00 | 0.043 / 1.00 | 0.307 / 0.86 |

## Temperature RMS error (K) after reconstruction

| Scenario | Oracle | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.92 | 9.39 | 0.90 | 1.33 | 1.21 | 0.98 | 0.98 | 13.53 |
| Plume, RAW stills | 0.92 | 9.35 | 0.96 | 1.12 | 0.99 | 0.76 | 0.76 | 13.79 |
| Plume, phone H.264 | 0.92 | 6.10 | 10.91 | 12.15 | 11.98 | 12.12 | 11.94 | 24.64 |
| Plume, 10-bit HEVC | 0.92 | 3.75 | 8.84 | 9.94 | 9.72 | 9.74 | 9.66 | 23.22 |
| Heated vertical plate (planar) | 0.14 | 1.36 | 0.80 | 0.37 | 0.42 | 1.32 | 0.27 | 68.19 |
| Camera shake | 0.92 | 10.93 | 1.10 | 1.48 | 1.26 | 0.71 | 0.63 | 13.65 |

## Runtime per frame (s)

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.08 | 0.13 | 0.27 | 0.30 | 0.34 | 11.17 | 1.55 |
| Plume, RAW stills | 0.08 | 0.24 | 0.06 | 0.25 | 0.26 | 5.60 | 0.12 |
| Plume, phone H.264 | 0.05 | 0.13 | 0.06 | 0.27 | 0.28 | 9.82 | 0.13 |
| Plume, 10-bit HEVC | 0.04 | 0.12 | 0.06 | 0.27 | 0.29 | 4.67 | 0.14 |
| Weak disturbance (breath), RAW | 0.06 | 0.13 | 0.07 | 0.31 | 0.24 | 4.57 | 0.23 |
| Weak disturbance (breath), H.264 | 0.06 | 0.13 | 0.06 | 0.28 | 0.14 | 4.43 | 0.18 |
| Turbulent multi-scale field | 0.04 | 0.12 | 0.07 | 0.27 | 0.28 | 7.66 | 0.18 |
| Weak lens (calibration object) | 0.28 | 0.35 | 0.29 | 0.49 | 0.68 | 3.74 | 0.45 |
| Heated vertical plate (planar) | 0.30 | 0.37 | 0.31 | 0.53 | 0.97 | 5.54 | 0.42 |
| Camera shake | 1.23 | 1.35 | 1.40 | 1.43 | 1.49 | 6.62 | 1.69 |
| Rolling shutter, unsteady plume | 0.05 | 0.12 | 0.06 | 0.27 | 0.27 | 8.52 | 0.20 |
| Display flicker banding | 0.76 | 0.84 | 0.88 | 0.96 | 1.00 | 6.30 | 0.93 |
| Display moiré | 0.05 | 0.12 | 0.06 | 0.28 | 0.28 | 5.41 | 0.20 |
| Random-dot background | 0.06 | 0.12 | 0.06 | 0.26 | 0.27 | 4.26 | 0.18 |
| Multi-scale noise background | 0.04 | 0.12 | 0.06 | 0.27 | 0.26 | 5.57 | 0.19 |

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
