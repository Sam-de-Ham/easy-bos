# Architecture

Easy BOS is a Python application: PySide6 for the UI, PyTorch for the GPU work, PyAV and
rawpy for input. All processing lives in `core/` and has no Qt dependency, so it runs and is
tested headless. The UI in `ui/` drives it through task threads.

## Code layout

```
src/easy_bos/
  app.py                entry point: high-DPI policy, theme, main window
  core/                 processing, no Qt
    physics.py          gas constants, Gladstone–Dale constant (Ciddor dispersion), ideal gas
    geometry.py         Camera, Display, Setup: sensitivity, blur, pattern sizing, moiré risk
    patterns.py         speckle, random dots, multi-scale noise, focus target; sRGB output
    media.py            sources: video (PyAV, 10-bit, keyframe seeking), image sequences, RAW
    preprocess.py       reference averaging, exposure and banding correction, stabilisation,
                        validity mask, automatic reference choice
    pipeline.py         Processor: frames -> corrected pairs -> displacement; quality presets
    quality.py          capture checks: exposure, sampling, focus, noise, banding, moiré, ...
    spline.py           prefiltered cubic and quintic B-spline interpolation (GPU)
    imops.py            tensor operations and finite differences
    solvers.py          preconditioned CG, FFT/DCT Neumann–Poisson solvers
    flow/               displacement estimators; registry and parameter metadata in base.py
    reconstruct/        integrate.py (WLS), abel.py (three-point), density.py (density, T)
    render.py           views: knife-edge, magnitude, direction, colour schlieren, ...
    results.py          HDF5 result store, read lazily per frame and cached
    export.py           PNG, TIFF, video, HDF5, CSV, PDF pattern
    figures.py          matplotlib figures for export
    project.py          .ebos project file (JSON): setup, background, capture, processing,
                        physics
    device.py, paths.py CUDA/CPU selection; per-user data and log directories
  bench/                synthetic benchmark
    fields.py           analytic refractive-index fields with exact 3-D truth (autograd)
    synth.py            physically based camera renderer with exact displacement truth
    suite.py, report.py scenarios and the generated report (docs/benchmark.md)
    tune.py             parameter sweeps (docs/tuning.md)
    cli.py, demo.py     command-line entry point; demo project
  ui/                   PySide6 application
    main_window.py      window, menus, page switching
    state.py            AppState and Task threads
    canvas.py           pyqtgraph image view with regions and profiles
    pages/              setup, capture, process and results pages
    pattern_window.py   pixel-exact full-screen pattern output
    theme.py, widgets.py, colorbar.py, frames.py, diagram.py
installer/              per-user installer around uv (install.ps1, .cmd wrapper, Inno Setup)
tests/                  pytest; test_ui.py is marked slow
```

The estimators are registered in `core/flow/base.py`: Variational flow, Cross-correlation,
Lucas–Kanade, Farnebäck, DIS, Wavelet flow and RAFT. The UI offers the first five. Wavelet
flow and RAFT are kept for the benchmark and library use.

## Conventions

- Displacement `D` satisfies `cur(x) = ref(x + D(x))`. It is in pixels, `u` to the right and
  `v` down, on the grid of the distorted (current) frame. Every estimator, the synthetic
  renderer and the reconstruction follow this.
- Deflection angle: `eps = D / Setup.pixels_per_radian`.
- Images entering an estimator are normalised to zero mean and unit standard deviation of the
  reference.
- The camera is focused on the background, so chief rays set the scale in the flow plane. The
  object-plane pixel size is `h = p·Z_A·(Z_B − f)/(f·Z_B)`.
- Processed results live on the cropped and binned grid. `results_page.processed_setup()`
  builds the matching Setup, with the pixel pitch multiplied by the binning factor.

## Processing pipeline

`core/pipeline.py` runs these steps for each frame:

1. Source. `media.py` opens video, image sequences or RAW. Frames are cropped to the region
   of interest and binned. Automatic binning keeps frames at most ~2200 px wide.
2. Reference. The flow-free frames at the start of the clip are chosen automatically
   (`preprocess.suggest_reference`), aligned if the camera moved, and averaged (mean or
   median). The reference can also be the previous frame.
3. Exposure. The current frame is scaled to the reference's mean intensity.
4. Camera motion. In auto mode a phase-correlation check decides whether the full motion
   estimate is needed. If motion is compensated, the reference is warped onto the frame.
5. Banding. Row gains are corrected against the motion-compensated reference.
6. Validity mask. Untextured areas and areas whose local brightness changed are excluded,
   combined with any user mask.
7. Estimator. The selected method returns `D` and, where available, a confidence map. Masked
   pixels are set to NaN. Optional temporal averaging over `k` frames follows.
8. Storage. `results.py` writes each frame to HDF5 as it completes. The next frame is decoded
   on a worker thread while the GPU works on the current one.
9. Presentation. `render.py` turns `D` into views. `reconstruct/` integrates and inverts it
   into density and temperature on demand.

## Design decisions

### Stack and distribution

- Python, PyTorch and PySide6. Python has the best ecosystem for GPU image processing, Qt gives
  pixel-exact full-screen pattern windows on any monitor, and the project stays in one
  language. Tauri was rejected for its IPC cost and harder pixel-exact output; Dear PyGui for
  its thin widget set. See `docs/research/stack.md`.
- PyTorch comes from the CUDA 13.0 wheel index and falls back to the CPU at runtime
  (`core/device.py`; `EASY_BOS_DEVICE=cpu` forces it).
- The installer bundles `uv.exe`, the source and `uv.lock`, and runs `uv sync --frozen` in the
  install folder. PyInstaller was rejected: a 3–4 GB onedir, antivirus false positives and
  CUDA DLL churn.

### Interpolation

- Warping uses prefiltered cubic B-splines (Unser 1999; Schreier et al. 2000). PyTorch's
  bicubic `grid_sample` iterates to a biased fixed point: a 0.3 px shift came out as 0.27 px.
  The B-spline gives 0.299 px, with analytic gradients consistent with the warp.
- Cubic B-spline sampling is done as four bilinear `grid_sample` lookups (Sigg & Hadwiger
  2005). The derivative weights keep a constant sign within each pair, so gradients are exact
  too. This replaced 16 gathers with float64 weights; consumer GPUs run float64 at a small
  fraction of float32 speed.

### Estimators

- Variational flow uses a soft curl penalty, not a potential parameterisation. A
  single-view BOS field is curl-free, but solving for a scalar potential made the CG systems
  badly conditioned.
- The variational solver is preconditioned with a DCT spectral solve (Makhoul DCT through a
  same-size FFT). It inverts the Neumann regulariser exactly; the data term enters only as its
  mean. Inner solves are inexact (tolerance 1e-2, at most 15 iterations). Warping stops when
  the mean update falls below 1e-3 px; the maximum update is dominated by pixels at a flame or
  hand that never settle. This took a full run from 6 s to 1.65 s per 1080p frame, with the
  benchmark unchanged and real frames within 0.0004 px mean of fully converged solves.
- Lucas–Kanade solves for the total displacement at each iteration. Accumulating increments
  drifted, and the error grew with the iteration count.
- Cross-correlation deforms only the reference, which keeps `D` on the `cur` grid. The
  symmetric PIV form would put `D` on the midpoint grid. The predictor update is low-pass
  filtered at window scale (Schrijer & Scarano 2008), and the median test ignores NaNs.
- Wavelet flow (wOFA) uses explicit synthesis matrices from PyWavelets (bior6.8,
  periodization) on a padded domain. The data term is one-sided, to keep the convention, with a
  Laplacian regulariser and coarse-to-fine unlocking of scales under L-BFGS. Two additions
  beyond the paper: blur continuation at coarse stages, which extends the capture range beyond
  5 px, and exclusion of samples outside the image. The loss is an unnormalised sum: a
  per-pixel normalised loss tripped the absolute stopping tolerance of torch's L-BFGS after
  one step, so whole stages did nothing. See `docs/research/wavelet-optical-flow.md`.
- Farnebäck input is rescaled to roughly 0–255. Its internal thresholds return near-zero flow
  on normalised data.
- Quality presets: Fast is coarse cross-correlation (16 px windows, 50 % overlap, 3 passes),
  Balanced is cross-correlation with defaults, Best is variational flow. On the GPU coarse
  correlation is both faster than DIS (0.08 against 0.24 s per 1080p frame) and more accurate
  in most scenarios, so DIS is available only under Custom. Wavelet flow and RAFT are not in
  the UI: neither gained anything on the benchmark, both took 10–50 times as long, and both
  misbehaved near a flame.

### Corrections

- Stabilisation is BOS-aware. A BOS field is curl-free, so camera rotation is the median curl
  divided by two. Translation is the dominant mode of the displacement distribution, found by
  annealed mean shift; a least-squares fit absorbed 0.016 px of plume signal on clean data.
  Motion is re-estimated iteratively, which fixed a 0.1 px under-estimate at 3 px shake.
  Reference frames are aligned before averaging. The default model is rigid (shift and
  rotation). Affine is offered only with a marked stable region, because the core of a plume
  mimics scaling.
- Auto stabilisation (the default) compensates only when the flow-free reference frames show
  camera motion (> 0.03 px), a stable region is marked, or a frame jumps (> 0.25 px). When flow
  fills the frame, a uniform shift cannot be told apart from camera motion: forced
  compensation absorbed 0.07 px of a turbulent field. A phase-correlation check of a few
  milliseconds skips the full estimate when the camera is steady and the frame did not jump.
- Banding correction runs after motion compensation. Row gains are the median ratio of
  horizontally smoothed rows. Measured before alignment, shifted texture looked like banding.
- Exposure matching is a ratio of mean intensities. A per-pixel regression collapsed (gain
  ~0.2) under a 3 px shake. Percentile or spread matching is biased, because an averaged
  reference is quieter than a single frame (variational EPE rose from 0.0025 to 0.024 px). The
  mean is unaffected by noise and alignment, and exposure changes are multiplicative.
- The automatic validity mask (on by default) excludes untextured areas, such as a monitor
  bezel, and areas whose smoothed brightness ratio leaves [0.7, 1.43], such as a flame, a
  lighter or a hand. Strong refraction alone moves a random texture's local mean by up to
  ~30 %, which sets the tolerance. Without the mask these areas produced 3–26 px spurious
  vectors.
- Automatic reference frames. Pressing record shakes a tripod for 1–2 s, as shifts and
  rolling-shutter shear of up to ~0.4 px; the first frames gave 0.08–0.25 px noise. The
  pipeline measures DIS motion between consecutive quarter-resolution frames over the first
  3 s and takes the quietest 16-frame window, which gave 0.011–0.020 px on real clips.
  Frame-difference statistics were rejected because they follow the codec's GOP, not the
  scene. Taking the first window that is steady enough was rejected because it lands in the
  decaying tail of the shake.

### Reconstruction

- Indirect Abel inversion is the default: WLS integration of the deflection to a projection,
  then three-point Abel inversion per row (Xiong et al. 2020). Direct inversion of the
  deflection is offered. The operators reproduce the reference errors in
  `docs/research/reconstruction.md` exactly.
- WLS integration fills masked or NaN pixels harmonically, so Abel rows through holes are not
  corrupted, and masks them again afterwards. The preconditioner is an unshifted FFT
  Neumann–Poisson solve. A shift comparable to the anchor strength stalled CG; a diagonally
  rescaled variant diverged at 100:1 weight contrast. Convergence is reported, with a warning
  when it fails.

### Input and capture checks

- Video frames are the decoder's luma plane, read directly. RGB conversion cost 160 ms per 4K
  frame; reading the plane costs 22 ms. Y is the only full-resolution channel of 4:2:0 video.
  Limited-range video (16–235) is rescaled to [0, 1] so the clipping checks work.
- Decoding is slice-threaded. Frame threading has to refill its pipeline after every seek:
  random access went from 625 ms to 175 ms with the same sequential throughput.
- The Setup page predicts moiré with a diffraction-MTF model of the imaged display grid and
  picks the integer display-pixel block size that gives elements of about 3 px on the sensor.
- Capture checks run on central crops at the processing resolution (about 30 s down to 6 s).
  The noise floor is reported at the processing resolution, in full-resolution pixels.
- The moiré statistic is the spectral peak over the median power at the same radius,
  normalised by ln N, so it does not depend on crop size. A local-median statistic missed the
  periodic structure in real monitor clips, which scored 170–700 against about 10 for a random
  pattern.
- Exposure clipping is a pile-up at the frame's extreme code value, measured on the textured
  background only. A fixed [0, 1] test misfired on real video: cameras record blacks below
  code 16, and a black bezel is not clipped shadow.

### Storage and display

- Results are stored as uncompressed float16. The quantisation step (< 3e-4 px) is well below
  the noise.
- Display goes through 8-bit conversion and lookup-table or OpenCV HSV colour maps: colour
  schlieren went from 310 to 55 ms per 1080p frame. The scrubber decodes on release.

## Performance

Measured on 4K H.264 clips (92 Mbit/s) processed at 1080p on an RTX 4070 Laptop GPU:

| Step | Time |
|---|---|
| Open clip | ~1 s |
| Automatic reference choice | ~6 s |
| Capture check | ~6 s |
| Preview one frame, Best | 1.8 s plus reference load |
| Full run, Best (variational) | 1.3 s per frame |
| Full run, Balanced (cross-correlation) | 0.30 s per frame |
| Full run, Fast (coarse cross-correlation) | 0.17 s per frame |

Decoding is not the bottleneck: software H.264 decoding of a 4K frame takes about 20 ms.

## Known limitations

- Consumer codecs are the main accuracy gap. On the synthetic H.264 and HEVC plume scenarios
  every method reaches only ~0.04 px EPE, which gives 9–12 K temperature RMS against a 0.9 K
  optics floor. Variational flow under-reads by about 14 % there (slope 0.86); DIS stays
  unbiased. Most of the error is low-frequency and follows the codec's block structure. The
  capture check flags compression; high bitrate or RAW is recommended.
- Faint concentric ripples (~0.02 px) appear in the displacement on real monitor clips where
  the moiré check reports strong periodicity. The likely cause is display–sensor interference
  (moiré, or temporal LCD effects). This is not yet confirmed.
- Farnebäck's large window spreads a flame's influence beyond the automatic mask, leaving a
  halo. The other methods are unaffected.
- Abel inversion assumes parallel rays; there is no perspective correction. Long focal lengths
  keep the error small, and the Setup page encourages them.
- Reconstructions saved with "All frames" are written to the results file but not reloaded
  with a project. Reconstruct again after opening.
- RAW input uses the two green Bayer sites at half resolution. The channel is fixed.
- WLS integration with uniform weights takes about 20 s per 1080p frame. A multigrid
  preconditioner would cut this; the default half resolution keeps it acceptable.
