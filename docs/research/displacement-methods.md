# Displacement estimation for BOS — research notes

Compiled 2026-11-01 from a sourced literature survey. [V] read in full, [A] abstract only, [U] unverified.

## Key sources
- Schmidt, Bathel, Grauer, Hargather, Heineck, Raffel, *Twenty-Five Years of BOS*, AIAA SciTech 2025
  (ntrs.nasa.gov/citations/20240014988) [V] — Sec. III compares algorithms.
- Atcheson, Heidrich, Ihrke 2009, Exp Fluids 46:467, doi:10.1007/s00348-008-0572-7 [V].
- Nicolas et al., wOFV, arXiv:2310.03980 [V].
- Molnar et al., physics-informed BOS, Exp Fluids 64:14 (2023), arXiv:2208.04280 [V].
- Shiba et al., event-based BOS, T-PAMI 46(4):2011, arXiv:2311.00434 + repo [V].
- Pocket Schlieren, Exp Fluids 66:147 (2025), arXiv:2404.16060 [V].

## Findings
- **Cross-correlation (CC):** robust and fast but sub-samples the field, underestimates peaks at
  sharp gradients and peak-locks; windows < 8×8 are unstable. Highest EPE in Atcheson
  (~0.1–0.25 px). BOS settings 8–16 px windows, 50 % overlap; multi-pass with deformation.
- **Lucas–Kanade / ILS (DIC style):** dense, 3×3–5×5 windows, gain/offset illumination model
  improves robustness (Mercier & Lacassagne 2023). Least parameter-sensitive (Atcheson).
- **Horn–Schunck / variational:** "with the right settings Horn-Schunck outperforms all other
  algorithms" (Atcheson), EPE < 0.1 px with multi-scale noise backgrounds; sensitive to α.
  Brox over-smooths. Farnebäck is the practical choice in Pocket Schlieren and the event-BOS
  pseudo-ground-truth.
- **Wavelet OF (wOFA/wOFV):** optimise wavelet coefficients of U with the nonlinear DFD data term
  `Σ [I0(x) − I1(x+U)]²` and Laplacian regulariser `λ Σ |ΔU|²`; coarse-to-fine by progressively
  unlocking finer detail bands (coarse coefficients stay free). Most accurate reported for BOS
  (NASA review, Molnar: better than HS); ~23 % below tuned PIV error for particles (Nicolas).
  Under-regularisation hurts sharply, over-regularisation gradually.
- **Curl-free:** single-view BOS displacement is the gradient of the projected index (∝ ∇∫n dz),
  so it is irrotational. Event-BOS's Poisson parameterisation v = ∇q cut AEE ~45 % vs direct flow.
  The curl of an estimate is a free quality diagnostic.
- **Learned flow (RAFT family):** no published sub-pixel BOS benchmark; FlowFormer/RIFE failed on
  real BOS (event-BOS). Stock RAFT works at 1/8 resolution with learned upsampling. Treat as an
  experimental comparison only; our benchmark decides.
- **Fourier demodulation** needs periodic backgrounds — not applicable to random/natural ones.
- Background quality matters as much as the algorithm: multi-scale noise made all OF methods more
  robust (Atcheson); binary random squares beat dots on phones (Pocket Schlieren, 25.9 vs 15.3 dB).
- Defocus blur often limits resolution more than the algorithm (NASA review).

## Method set in Easy BOS
See `docs/architecture.md` for the reasons behind each choice.

1. Reference difference: instant qualitative view ("Quick look").
2. OpenCV Farnebäck and DIS: fast dense baselines, as used by Pocket Schlieren.
3. GPU FFT cross-correlation, multi-pass with window deformation: PIV-style method. The
   "Balanced" preset, and with 16 px windows and 3 passes the "Fast" preset.
4. Pyramidal dense Lucas–Kanade with local gain and offset.
5. Variational optical flow (warped, Charbonnier data term, Horn–Schunck or Laplacian
   regulariser) with a soft penalty on the curl of the field: the "Best" preset and the
   default. A potential (v = ∇q) parameterisation was tried and dropped because it made the
   linear systems badly conditioned.
6. Wavelet optical flow (wOFA) and 7. RAFT (torchvision, off-the-shelf): kept for the
   benchmark and library use, not offered in the UI. Neither improved on variational flow in
   the synthetic benchmark, both took 10–50 times as long, and both misbehaved near a flame.
