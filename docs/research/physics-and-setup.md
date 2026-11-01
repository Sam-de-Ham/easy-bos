# BOS physics, geometry and setup — research notes

Compiled 2026-11-01 from a sourced literature review. Tags: **[V]** read in the primary
source, **[S]** from abstract/secondary, **[D]** derived here and checked numerically,
**[U]** unverified judgement.

## 1. Image formation

Paraxial, small-angle (Raffel 2015, Exp Fluids 56:60, doi:10.1007/s00348-015-1927-5, Eqs. 1–4 [V]).

Distances: `Z_A` lens→object, `Z_D` object→background, `Z_B = Z_A + Z_D`. Camera focused
on the background.

- Deflection: `ε = (1/n0) ∫ ∂n/∂y dz` [rad]
- Sensor displacement: `Δy = f · Z_D / (Z_B − f) · ε`
- In pixels (pitch `p`): `Δpx = f·Z_D / ((Z_B − f)·p) · ε` [D]
- Object-plane pixel size (use for gradients and Poisson grids): `h = p·(Z_A − f)/f` [D]
- Background magnification: `M = f/(Z_B − f)`

### Blur / aperture (Raffel Eqs. 6–8 [V])
- Light-cone footprint at the object (sets resolution): `b_obj = (f/N)·Z_D/Z_B` [D]
- Diffraction dot image: `d_d = 2.44·N·(M+1)·λ`
- Default f/8–f/11. Choose `N ≥ f·Z_D/(Z_B·b_max)` and `N ≤ D_dot/(2.44 λ (1+M))`.

Accuracy: dynamic range ≈ 50:1, typical error 2–3 % full scale, displacements often < 0.5 px.

## 2. Reconstruction

- Gladstone–Dale: `n − 1 = K ρ`, `K = 2.26e-4 m³/kg` (Ciddor dispersion gives 2.290e-4 @450 nm,
  2.268e-4 @550 nm, 2.257e-4 @633 nm) [D]. CO₂/H₂O change K by a few % [U].
- `ρ_amb = p·M_air/(R·T)`; `M_air = 0.028965 kg/mol`, `R = 8.31446 J/(mol K)`.
- Temperature at constant pressure and composition: `T = T_amb·ρ_amb/ρ`.

### Planar field of known depth W
`∇ρ = n0 · Δx_sensor / (M · Z_D · K · W)`, on grid spacing `h`. Integrate with Poisson
(Richard & Raffel 2001) or **weighted least squares** `ρ = (GᵀWG)⁻¹ GᵀW g` with
`W = σ⁻²` from displacement uncertainty (Rajendran et al., arXiv 2004.01217 [V]: ~80 %
lower random error than Poisson in synthetic tests). Needs ≥1 Dirichlet point (ambient).

### Axisymmetric
`δ(r) = n/n0 − 1`, projection `P(y) = 2∫_y^R δ r/√(r²−y²) dr`, `ε(y) = dP/dy`.
- Direct: `δ(r) = −(1/π) ∫_r^R ε(y)/√(y²−r²) dy`.
- Indirect (more noise-robust, Xiong et al. 2020 doi:10.1007/s00348-020-03007-4 [S]):
  integrate ε → P, then Abel-invert P; three-point Abel (Dasch 1992) best with noise.
- Onion peeling: `W_ji = 2[√(r_{i+½}² − y_j²) − √(max(r_{i−½}, y_j)² − y_j²)]` + Tikhonov.
- Axis: maximise antisymmetry of ε; average the two halves.
- Perspective (non-parallel rays) causes errors (Sipkens 2022 [S]); keep the field angle small.

## 3. Background patterns
- Dot image 3 px on sensor (2.5–4) [V Raffel; S Mercier & Lacassagne 2023]; random positions
  with min-distance; full contrast.
- Physical dot size `s = D_px·p/M`.
- Multi-octave/wavelet noise beats block correlation with optical flow (Atcheson 2009 [S]);
  Pocket Schlieren recommends wavelet noise for smartphones [S].
- Displays [D]: imaged display pitch `M·p_disp`. Safe if beyond optical cutoff
  (`M·p_disp < λN`) or ≥ 4 camera px per display px. Avoid 0.5–3 px unless defocused.
  Draw elements as integer blocks of ≥ 2–3 display pixels, greyscale.
- PWM/mains flicker + rolling shutter → horizontal banding. Detect from row-mean
  spectrum; normalise row gain. Exposure = integer multiple of 10 ms (50 Hz) / 8.33 ms (60 Hz).
- Print: native printer DPI, dot ≥ 3–4 printer dots, matte, fiducials/scale bar.
- Background quality: displacement noise std from two no-flow frames (Bauknecht 2014 [V via Raffel]).

## 4. Camera
- No BOS study quantifies H.264/HEVC error — benchmark it ourselves.
- Stabilisation (IBIS/OIS/EIS) must be off; manual focus on background; AE/AF locked; base ISO.
- Reference: average 16–32 no-flow frames.
- Drift: fit global affine/homography on a flow-free region (RANSAC) and subtract.

## 5. Expected magnitudes [D]
Axisymmetric Gaussian `Δn = A·exp(−r²/2σ²)`: `|ε|max = √(2π)e^{−½}·A = 1.52·A` at `y = σ`.
Candle plume (~100 K), f = 50 mm, Z_A = Z_D = 1 m, 4 µm pixels → ≈ 0.67 px peak.
Breath (~10 K) ≈ 0.1 px. Target noise floor ≈ 0.02 px.

Calibration objects: weak lens of focal F gives `ε = r/F` (radial, linear);
ophthalmic prism gives uniform `ε`; flat window = null test; heated vertical plate = planar case.

## 6. Synthetic benchmark guidance
Rajendran, Bane & Vlachos 2019 (doi:10.1088/1361-6501/ab1ca8) ray-trace; benchmark Gaussian
density field, 3 px dots, 15 dots / 32², 1 % noise. Cheaper exact route: analytic displacement,
band-limited resampling (no bilinear), oversampled render + PSF + pixel integration,
Poisson shot + read noise + PRNU + quantisation, optional real codec round-trip,
row-gain banding, rigid shake. Metrics: endpoint error, angular error, density/temperature RMS.
