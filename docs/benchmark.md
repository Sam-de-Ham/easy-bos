# Synthetic benchmark

Generated 2026-11-01 by `uv run easy-bos-bench` on NVIDIA GeForce RTX 4070 Laptop GPU · 8 GB · CUDA, 512×384 px frames. Every method runs through the production pipeline with its default parameters. Displacement errors are endpoint errors in pixels over the frame minus a 16-px border; *rel* is mean EPE divided by the mean true magnitude; *peak* is the ratio of the 99th-percentile magnitudes (1 = no attenuation). Temperature error is the RMS over the region more than 20 % of the peak rise above ambient. *Oracle* reconstructs from the true (depth-of-field blurred) displacement: its error is the floor set by optics and reconstruction, not by the estimator.

## Mean endpoint error (px), lower is better

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.0320 | 0.0183 | 0.0160 | **0.0160** | 0.0160 | 0.0160 | 0.0478 |
| Plume, RAW stills | 0.0322 | 0.0192 | 0.0162 | 0.0188 | **0.0161** | 0.0162 | 0.0474 |
| Plume, phone H.264 | **0.0407** | 0.0475 | 0.0449 | 0.0513 | 0.0430 | 0.0458 | 0.0618 |
| Plume, 10-bit HEVC | **0.0354** | 0.0430 | 0.0405 | 0.0490 | 0.0392 | 0.0414 | 0.0613 |
| Weak disturbance (breath) | 0.0135 | 0.0135 | 0.0134 | 0.0136 | 0.0134 | **0.0133** | 0.0157 |
| Turbulent multi-scale field | 0.0399 | 0.0310 | 0.0274 | 0.0284 | 0.0278 | **0.0267** | 0.1250 |
| Weak lens (calibration object) | 4.8174 | 5.9053 | 8.5225 | 17.8165 | 18.0409 | 17.9436 | **1.1061** |
| Heated vertical plate (planar) | 0.0470 | 0.0507 | 0.0456 | 0.0130 | 0.1918 | **0.0119** | 0.4347 |
| Camera shake | 0.1848 | 0.2179 | 0.0826 | 0.3678 | **0.0651** | 0.0775 | 0.2402 |
| Display flicker banding | 0.0322 | 0.0193 | 0.0162 | 0.0189 | **0.0162** | 0.0163 | 0.0475 |
| Display moiré | 0.0378 | 0.0207 | 0.0177 | 0.0215 | 0.0174 | **0.0173** | 0.0444 |
| Random-dot background | 0.0402 | 0.0195 | 0.0165 | 0.0186 | 0.0164 | **0.0163** | 0.0495 |
| Multi-scale noise background | 0.0264 | 0.0213 | 0.0168 | 0.0246 | **0.0163** | 0.0167 | 0.0422 |

## Relative error and peak recovery

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.241 / 1.12 | 0.138 / 1.01 | 0.121 / 1.00 | 0.120 / 0.99 | 0.120 / 0.98 | 0.120 / 1.00 | 0.360 / 0.82 |
| Plume, RAW stills | 0.242 / 1.12 | 0.145 / 1.02 | 0.122 / 1.00 | 0.141 / 1.00 | 0.121 / 0.99 | 0.122 / 1.00 | 0.357 / 0.80 |
| Plume, phone H.264 | 0.307 / 0.99 | 0.358 / 0.98 | 0.338 / 0.95 | 0.387 / 0.99 | 0.324 / 0.91 | 0.345 / 0.96 | 0.466 / 0.59 |
| Plume, 10-bit HEVC | 0.267 / 1.00 | 0.324 / 0.95 | 0.305 / 0.91 | 0.369 / 0.98 | 0.295 / 0.88 | 0.312 / 0.92 | 0.462 / 0.64 |
| Weak disturbance (breath) | 1.016 / 0.13 | 1.015 / 0.16 | 1.009 / 0.07 | 1.025 / 0.27 | 1.007 / 0.07 | 1.000 / 0.00 | 1.185 / 0.66 |
| Turbulent multi-scale field | 0.199 / 0.99 | 0.154 / 1.00 | 0.136 / 1.01 | 0.141 / 1.01 | 0.138 / 0.98 | 0.133 / 1.01 | 0.622 / 0.76 |
| Weak lens (calibration object) | 0.267 / 0.95 | 0.327 / 0.83 | 0.472 / 1.09 | 0.988 / 0.12 | 1.000 / 0.02 | 0.995 / 0.06 | 0.061 / 1.02 |
| Heated vertical plate (planar) | 0.105 / 0.95 | 0.113 / 0.98 | 0.102 / 1.00 | 0.029 / 1.00 | 0.428 / 0.53 | 0.027 / 0.99 | 0.970 / 0.08 |
| Camera shake | 1.391 / 1.26 | 1.640 / 1.63 | 0.622 / 1.18 | 2.769 / 3.70 | 0.490 / 1.05 | 0.584 / 1.18 | 1.808 / 0.82 |
| Display flicker banding | 0.242 / 1.12 | 0.146 / 1.02 | 0.122 / 1.00 | 0.142 / 1.00 | 0.122 / 0.99 | 0.123 / 1.00 | 0.358 / 0.83 |
| Display moiré | 0.291 / 1.14 | 0.160 / 1.03 | 0.137 / 1.01 | 0.166 / 1.03 | 0.134 / 0.99 | 0.133 / 1.00 | 0.342 / 0.85 |
| Random-dot background | 0.303 / 1.16 | 0.147 / 1.02 | 0.124 / 1.01 | 0.140 / 1.00 | 0.123 / 0.99 | 0.123 / 1.00 | 0.373 / 0.80 |
| Multi-scale noise background | 0.199 / 1.08 | 0.161 / 1.02 | 0.127 / 1.01 | 0.185 / 1.03 | 0.123 / 0.99 | 0.126 / 1.01 | 0.318 / 0.86 |

## Temperature RMS error (K) after reconstruction

| Scenario | Oracle | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 0.92 | 9.43 | 1.18 | 1.25 | 1.46 | 1.71 | 1.23 | 12.58 |
| Plume, RAW stills | 0.92 | 9.43 | 1.25 | 1.14 | 1.32 | 1.52 | 1.13 | 12.83 |
| Plume, phone H.264 | 0.92 | 5.72 | 10.81 | 11.52 | 11.89 | 11.58 | 11.46 | 24.36 |
| Plume, 10-bit HEVC | 0.92 | 4.36 | 9.67 | 10.21 | 10.85 | 10.47 | 10.39 | 23.53 |
| Heated vertical plate (planar) | 0.14 | 1.34 | 0.37 | 1.08 | 0.21 | 30.64 | 0.27 | 68.26 |
| Camera shake | 0.93 | 17.41 | 9.86 | 2.53 | 10.16 | 3.56 | 2.54 | 29.77 |

## Runtime per frame (s)

| Scenario | DIS | Farnebäck | Cross-correlation | Lucas–Kanade | Variational flow | Wavelet flow | RAFT (experimental) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plume, ideal camera | 1.88 | 0.67 | 0.74 | 1.20 | 1.48 | 11.40 | 2.19 |
| Plume, RAW stills | 1.00 | 1.44 | 0.75 | 1.39 | 3.78 | 9.51 | 1.22 |
| Plume, phone H.264 | 1.57 | 2.06 | 1.32 | 1.70 | 2.24 | 6.86 | 1.21 |
| Plume, 10-bit HEVC | 0.96 | 0.96 | 0.82 | 1.61 | 2.36 | 6.57 | 1.27 |
| Weak disturbance (breath) | 1.22 | 1.17 | 1.25 | 1.80 | 3.27 | 1.34 | 2.27 |
| Turbulent multi-scale field | 1.42 | 1.03 | 1.10 | 1.25 | 2.11 | 5.15 | 1.05 |
| Weak lens (calibration object) | 0.81 | 0.93 | 0.67 | 1.38 | 3.48 | 9.15 | 1.21 |
| Heated vertical plate (planar) | 0.76 | 0.80 | 0.97 | 1.98 | 3.95 | 9.80 | 0.88 |
| Camera shake | 0.62 | 1.04 | 1.18 | 1.74 | 5.22 | 14.43 | 1.44 |
| Display flicker banding | 1.31 | 1.29 | 1.39 | 2.28 | 4.14 | 8.89 | 1.01 |
| Display moiré | 0.75 | 0.62 | 0.66 | 1.25 | 1.63 | 5.00 | 0.84 |
| Random-dot background | 0.70 | 0.86 | 1.01 | 2.03 | 4.10 | 9.53 | 1.72 |
| Multi-scale noise background | 1.63 | 1.28 | 1.38 | 1.92 | 2.27 | 5.86 | 0.78 |

## Scenarios

- **Plume, ideal camera** (`plume-ideal`): Axisymmetric plume (~+75 K), noise-free 16-bit sensor: algorithmic accuracy.
- **Plume, RAW stills** (`plume-raw`): 14-bit linear sensor with shot/read noise and PRNU, 4 averaged references.
- **Plume, phone H.264** (`plume-h264`): 8-bit BT.709 video, x264 CRF 20, phone-like noise and softer optics.
- **Plume, 10-bit HEVC** (`plume-hevc10`): 10-bit BT.709 video, x265 CRF 22 (mirrorless S-log/HLG-style capture).
- **Weak disturbance (breath)** (`breath`): ~+8 K plume: peak displacement ~0.04 px, 8-bit H.264. Tests the noise floor.
- **Turbulent multi-scale field** (`turbulent`): Random eddies 2-12 mm at f/16 and short Z_D (little blur): spatial resolution.
- **Weak lens (calibration object)** (`lens`): Thin lens F = 3 m: radial displacement linear in radius, up to ~1.5 px.
- **Heated vertical plate (planar)** (`plate`): 2-D natural-convection boundary layer, 15 cm deep: planar reconstruction.
- **Camera shake** (`shake`): Plume with 0.8 px rms shake and 0.05 deg rotation per frame; stabilisation on.
- **Display flicker banding** (`banding`): 3 % PWM row banding seen through a rolling shutter; banding correction on.
- **Display moiré** (`moire`): Background on a display whose pixel grid images to 1.4 camera px at f/4.
- **Random-dot background** (`dots`): Same RAW plume with a sparse random-dot pattern instead of speckle.
- **Multi-scale noise background** (`multiscale`): Same RAW plume with a multi-scale noise pattern.
