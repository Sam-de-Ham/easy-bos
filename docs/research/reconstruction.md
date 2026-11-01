# Density reconstruction — verified specifications

Compiled 2026-11-01. Sources read in full: PyAbel `abel/dasch.py` + docs (MIT); Xiong, Kaufmann &
Noiray, Exp Fluids 61:178 (2020); Rajendran et al. arXiv:2004.01217 (published
doi:10.1007/s00348-020-03071-w); Zhang et al. GLS arXiv:1912.00130; Molnar et al. arXiv:2208.04280.
Dasch 1992 itself (paywalled) via PyAbel/Xiong reproductions; Kolhe & Agrawal 2009 abstract only.

## Three-point Abel (Dasch 1992, as corrected by Martin 2002)
Index units, `y_j = jΔr`, `j = 0` on the axis. `f_i = (1/Δr) Σ_j D_ij P_j`.

I⁰_ij = 0 (j < i or i = j = 0);
(1/2π) ln[(√((2j+1)²−4i²) + 2j+1)/(2j)] (j = i ≠ 0);
(1/2π) ln[(√((2j+1)²−4i²) + 2j+1)/(√((2j−1)²−4i²) + 2j−1)] (j > i).

I¹_ij = 0 (j < i); (1/2π)√((2j+1)²−4i²) − 2j I⁰_ij (j = i, incl. I¹₀₀ = 1/2π);
(1/2π)[√((2j+1)²−4i²) − √((2j−1)²−4i²)] − 2j I⁰_ij (j > i).

D_ij: 0 (j < i−1); I⁰_{i,j+1} − I¹_{i,j+1} (j = i−1); I⁰_{i,j+1} − I¹_{i,j+1} + 2I¹_{ij} (j = i);
I⁰_{i,j+1} − I¹_{i,j+1} + 2I¹_{ij} − I⁰_{i,j−1} − I¹_{i,j−1} (j ≥ i+1);
special case D_01 = I⁰₀₂ − I¹₀₂ + 2I¹₀₁ − 2I¹₀₀. P = 0 beyond the last sample.

## Direct inversion from the deflection
`f(r) = −(1/π)∫_r^R ε/√(y²−r²) dy`, same quadrature:
`f_i = −Σ_{j≥i}[2 I⁰_ij ε_j + ½ I¹_ij (ε_{j+1} − ε_{j−1})]`, with `ε_0 = 0`, `ε_{−1} = −ε_1`,
`ε_N = 0`. No 1/Δr factor. Most accurate in noise-free tests (max err 2.5e-4 at N = 101 on a
Gaussian); noise gain ≈ indirect in 1-D.

## Indirect route (recommended for noisy data; Xiong et al.)
Solve the 2-D Poisson / least-squares problem for the projection P from both εx and εy, then
three-point Abel per row. ≈ 2.5× lower error than direct at ≥ 2 % noise.

## Weighted least squares integration (Rajendran et al.)
`ρ = (GᵀWG)⁻¹ GᵀW g`, `W_k = σ_k⁻²` from the displacement uncertainty.
Staggered-grid G (rows `(ρ_{i+1} − ρ_i)/Δx` at midpoints), measured gradients averaged to the
midpoints, edge weight `1/(¼(σ_k² + σ_{k+1}²))`. Dirichlet anchors as extra weighted rows
`Lρ = ρ_ref` (≥ 1 per connected region). Masks: drop edges touching invalid nodes. Direct sparse
solve or CG. WLS: −50 % RMS random error (synthetic), −30 % uncertainty (experiment) vs Poisson.

## Axis and boundaries
- Axis: minimise `Σ[ε(c+y) + ε(c−y)]²` over c per row (or argmin of autoconvolution), sub-pixel
  by parabola; resample and antisymmetrise `ε_s(y) = ½[ε(c+y) − ε(c−y)]`.
- All inversions assume P = ε = 0 beyond R; a truncated tail biases f near R → keep the field of
  view wide, or trust r ≲ 0.8R and quote relative to the reference at R.
