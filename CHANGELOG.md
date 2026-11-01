# Changelog

## 1.1.0 — 2026-11-01

Faster and simpler, after the first round of real footage (Sony A6400, 4K H.264, a lighter
in front of a 27" monitor).

- Processing is several times faster. Frames are read straight from the video's luma plane,
  cubic B-spline sampling runs as four bilinear GPU lookups, and variational flow uses a DCT
  preconditioner with inexact inner solves. On 4K footage processed at 1080p: 0.17 s per
  frame (Fast), 0.30 s (Balanced), 1.3 s (Best).
- Reference frames are chosen automatically: the steadiest stretch in the first 3 s, after
  the shake from pressing record. Noise floor on the test clips went from 0.08–0.25 px to
  0.011–0.020 px.
- Automatic validity mask: untextured areas (a monitor bezel) and anything that glows or
  blocks the background (a flame, a lighter, a hand) are excluded instead of producing
  spurious vectors.
- Quality presets on the Process page (Fast, Balanced, Best, Custom); secondary settings
  folded away; a time estimate after a preview.
- Method comparison is a steady grid with a shared scale; click a result to use that method.
- Wavelet flow and RAFT are no longer offered in the app. They stay in the library and the
  benchmark.
- 4K and larger frames are binned automatically to at most about 2200 px wide.
- Capture check: about 25 s → 6 s. New moiré statistic that does not depend on crop size.
  Exposure clipping is detected as a pile-up at the extreme code values, so video footroom
  and dark bezels no longer read as clipped.
- Setup page: a rule of thumb for distances and hover hints with targets for every input;
  1440p monitor presets.
- Scrubbing decodes on release; colour views render through lookup tables.
- Fixed: a crash when the app was started without a console (from the Start-menu shortcut)
  and a library printed progress.

## 1.0.0 — 2026-11-01

First version.

- Setup page: sensitivity, resolution and blur for a camera, lens and distances; background
  pattern sizing with a moiré prediction; full-screen pattern display on any monitor and PDF
  export for printing.
- Capture: video (including 10-bit HEVC), image sequences and RAW; capture check for
  exposure, sampling, focus, noise floor, stability, flicker banding, moiré and compression.
- Processing on the GPU (CUDA) with a CPU fallback: variational optical flow, multi-pass
  cross-correlation, Lucas–Kanade, Farnebäck, DIS, wavelet optical flow and RAFT. Camera
  motion, exposure drift and display flicker are corrected.
- Results: knife-edge, colour schlieren, magnitude, direction, components and divergence;
  density and temperature through Abel inversion or a planar model; profiles, time series and
  histograms; export to PNG, TIFF, video, HDF5, CSV and figures.
- Synthetic benchmark with exact ground truth over 15 scenarios, and a demo project.
- Per-user Windows installer.
