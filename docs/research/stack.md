# Technology stack — decision notes (2026-11-01)

- **Python 3.12** (av 19, numpy 2.5, scipy 1.18 require ≥ 3.12), managed by uv.
- **PySide6-Essentials 6.12** UI; Fusion style + custom dark QPalette + small QSS.
  Rejected Tauri/web (two languages, IPC cost for frames, harder pixel-exact fullscreen) and
  Dear PyGui (thin widget set).
- **pyqtgraph 0.14** for interactive images/plots; matplotlib only for offscreen figure export.
- **Pattern window 1:1 pixels:** `HighDpiScaleFactorRoundingPolicy.PassThrough` before
  QApplication; render the pattern at `geometry × devicePixelRatio` and set the image DPR.
- **PyTorch 2.14.1 + torchvision 0.29.1 on the cu130 index** (released pair). The CUDA wheel
  falls back to CPU at runtime. `tool.uv.sources` is not carried into wheel metadata, so the
  app ships with `uv.lock` and installs with `uv sync --frozen`.
- **RAFT weights** `raft_large_C_T_SKHT_V2` (20 MB) download on first use into
  `%LOCALAPPDATA%\EasyBOS\models` (`TORCH_HOME`); never in git. RAFT is available for the
  benchmark and library use only, not in the UI. Weights are
  trained on research datasets (FlyingChairs/Things): flag for commercial users.
- **PyAV 19** (bundled FFmpeg; HEVC Main10 decode, libx264/x265/prores_ks/ffv1/nvenc encode).
  Do not use OpenCV video I/O (8-bit only, duplicate FFmpeg DLLs).
- **rawpy 0.28** (LibRaw 0.22) for ARW/DNG; linear 16-bit output.
- **opencv-contrib-python-headless 5.0** for DIS / Farnebäck and image utilities.
- **h5py** for raw data export (self-describing, readable in MATLAB/ParaView).
- **Distribution:** a per-user installer (zip with a PowerShell script, or Inno Setup `.exe`) that bundles `uv.exe` + source + `uv.lock` and runs
  `uv sync --frozen` into `%LOCALAPPDATA%\EasyBOS` (the ComfyUI-Desktop / InvokeAI pattern).
  Rejected PyInstaller (3–4 GB onedir, AV false positives, CUDA DLL churn), Nuitka (fragile
  with torch), Briefcase (no custom indexes).

Windows pitfalls: old drivers silently mean CPU (the main window shows the compute device); keep the install out of
OneDrive; `UV_LINK_MODE=copy`; Night Light / HDR / auto-brightness change pattern intensities;
hybrid laptops need "High performance" GPU preference.
