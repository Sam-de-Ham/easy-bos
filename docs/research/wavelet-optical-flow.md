# Wavelet optical flow (wOFV / wOFA) — verified specification

Compiled 2026-11-01. Read in full: Nicolas et al. arXiv:2310.03980; PIVlab `+wOFV/` source
(Jassal & Schmidt) and its OSF filter matrices; PyTyphoon (Dérian). Abstract only: Schmidt &
Sutton 2019/2020/2021, Schmidt & Woike 2021 (BOS, "wOFA"). Not obtainable: Kadri-Harouna 2013,
Dérian 2013 (HAL bot wall). [V-text] full text, [V-code] code, [A] abstract, [I] inferred.

- **Unknown:** `U_k = Ψ θ_k`, separable 2-D DWT synthesis, Mallat layout, full depth [V-code].
  Square power-of-2 tiles in PIVlab, symmetric boundary [V-code].
- **Finest scale never estimated:** finest detail coefficients stay zero — implicit
  regularisation by truncation (Schmidt & Sutton 2021) [V-code].
- **Data term:** Nicolas: `Σ[I0(x) − I1(x+U)]²` (non-linearised) [V-text]; current PIVlab code is
  symmetric `I0(x − U/2) − I1(x + U/2)` [V-code]. Spline interpolation.
- **Regulariser:** Laplacian `λ Σ |∇²U_k|²` (Nicolas; Schmidt & Woike per JORS) or viscous
  `½|∇²U + ⅓∇(∇·U)|²` (PIVlab default) [V].
- **Schedule:** for s = coarsest … finest−1, optimise the whole top-left 2^s block (all coarser
  coefficients free), warm start from the previous stage. No image pyramid in the original;
  PIVlab adds an optional 3-level pyramid for large displacements [V-code].
- **Solver:** L-BFGS, memory 8, ≤ 100 iterations / 300 evaluations per stage, tol 1e-5 [V-code].
  Both reference codes map the gradient with F instead of F⁻ᵀ (only exact for orthogonal
  wavelets); autograd gives the true gradient [V-code, measured].
- **Wavelet:** BNC 17/11 (Nicolas), bior6.8 = CDF 17/11 (PIVlab) [V].
- **λ:** not transferable between datasets. Images in [0, 1]; Nicolas λ 0.1–1 experimental,
  100–180 synthetic 8-bit; under-regularisation hurts sharply, over-regularisation gradually [V-text].
- **Accuracy:** ~23 % below tuned PIV (Nicolas); 0.194 vs 0.222 RMSE vs CC on 3-D HIT (JORS);
  BOS: beats ILS when the field has high-frequency content [A].
- **Curl-free wavelet variant:** none published.

## Our implementation choices
- Synthesis operator built from `pywt` bior6.8 reconstruction filters, implemented in torch
  with periodic extension on a domain padded to a multiple of 2^L; the padding margin carries no
  data weight, so wrap-around never meets measured pixels. Verified against `pywt.waverec2`
  (periodization) in tests.
- Symmetric data term with the prefiltered cubic B-spline (`core/spline.py`), true gradients via
  autograd, `torch.optim.LBFGS` with strong-Wolfe line search.
- Laplacian regulariser with a finite-difference stencil on the reconstructed field.
