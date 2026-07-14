"""DINEOF (Data Interpolating Empirical Orthogonal Functions) implementation.

Reference: Beckers & Rixen (2003), "EOF Calculations and Data Filling from
Incomplete Oceanographic Datasets."

Standard gold-standard baseline for SST gap-filling in the oceanographic
community. Iterative SVD reconstruction with truncation at k modes.

Algorithm
---------
Given a data matrix X (T × P) with some entries missing:
  1) Initialise missing entries with the temporal mean of each column
     (or 0 if a column is fully missing).
  2) Compute the truncated SVD: X ≈ U_k Σ_k V_k^T  (top-k modes).
  3) Replace missing entries with the reconstruction X_hat.
  4) Repeat 2–3 until ‖X^(it) − X^(it−1)‖ / ‖X^(it−1)‖ < tol.
  5) (optional) cross-validate on a held-out subset of the observed entries
     to pick the optimal k. We use a fixed k by default since the 30-day
     window is short.
"""
import numpy as np


def dineof_fill(data, valid, *, k=5, max_iter=30, tol=1e-4, verbose=False,
                center=True):
    """Fill the missing entries (valid == 0) of `data` via iterative SVD.

    Args:
        data:  (T, H, W) float32, with NaN allowed at missing positions
        valid: (T, H, W) uint8/bool, 1 where data is the original observation
               we trust (the "anchor" set). 0 where DINEOF must impute.
        k:     number of EOF modes to retain.
        max_iter: max iterations of SVD reconstruction.
        tol:   relative-change convergence threshold.
        verbose: print iteration progress.
        center: subtract per-pixel temporal mean before SVD (standard DINEOF
                preprocessing; SVD now models DEVIATIONS not absolute SST).

    Returns:
        filled: (T, H, W) float32, observed pixels untouched; missing pixels
                filled by iterative SVD reconstruction.
    """
    assert data.shape == valid.shape, "shape mismatch"
    T, H, W = data.shape
    X = data.reshape(T, -1).astype(np.float64)   # (T, P)
    V = valid.reshape(T, -1).astype(bool)         # (T, P)

    # Column-wise temporal mean over OBSERVED entries
    col_sum = np.where(V, X, 0).sum(axis=0)        # (P,)
    col_cnt = V.sum(axis=0)                        # (P,)
    global_mean = float(X[V].mean()) if V.any() else 0.0   # observed grand mean
    col_mean = np.where(col_cnt > 0,
                        col_sum / np.maximum(col_cnt, 1),
                        global_mean)               # (P,) never-obs -> grand mean, not 0K

    # Init: keep observed values; fill missing with column mean
    X_filled = np.where(V, X, np.broadcast_to(col_mean, X.shape))

    # Optional centering: SVD on (X - col_mean), then add col_mean back
    if center:
        X_centered = X_filled - col_mean[None, :]
    else:
        X_centered = X_filled

    prev = X_centered.copy()
    for it in range(max_iter):
        # Truncated SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        k_eff = min(k, len(s))
        recon = (U[:, :k_eff] * s[:k_eff]) @ Vt[:k_eff, :]

        # Build "back in original space" for the observed comparison
        if center:
            recon_orig = recon + col_mean[None, :]
        else:
            recon_orig = recon

        # Replace missing entries with reconstruction; keep observed entries
        X_filled = np.where(V, X, recon_orig)

        # Update centered version for next iter
        if center:
            X_centered = X_filled - col_mean[None, :]
        else:
            X_centered = X_filled

        num = np.linalg.norm(X_centered - prev)
        den = np.linalg.norm(prev) + 1e-12
        rel = num / den
        if verbose:
            print(f"  DINEOF k={k_eff} iter {it+1}: rel={rel:.3e}")
        if rel < tol:
            break
        prev = X_centered

    return X_filled.reshape(T, H, W).astype(np.float32)


def dineof_with_cv(data, valid, *, k_candidates=(2, 4, 6, 8, 10, 15),
                   max_iter=20, tol=1e-3, cv_frac=0.05, seed=0):
    """DINEOF with cross-validation to pick k.

    Holds out cv_frac of observed entries, runs DINEOF with each k, picks
    the k that minimises reconstruction RMSE on the held-out entries.
    """
    rng = np.random.default_rng(seed)
    V = valid.astype(bool)
    flat_obs = np.where(V.ravel())[0]
    n_hold = max(1, int(len(flat_obs) * cv_frac))
    hold_idx = rng.choice(flat_obs, size=n_hold, replace=False)
    hold_mask_flat = np.zeros(V.size, dtype=bool)
    hold_mask_flat[hold_idx] = True
    hold_mask = hold_mask_flat.reshape(V.shape)

    # Hide held-out entries from "valid"
    valid_cv = V & ~hold_mask
    true_held = data[hold_mask]

    best_k, best_rmse, best_recon = None, np.inf, None
    for k in k_candidates:
        recon = dineof_fill(data, valid_cv, k=k, max_iter=max_iter, tol=tol)
        rmse = float(np.sqrt(np.mean((recon[hold_mask] - true_held) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k
            best_recon = recon

    # Final run with the chosen k on FULL valid set
    final = dineof_fill(data, V, k=best_k, max_iter=max_iter, tol=tol)
    return final, best_k, best_rmse


if __name__ == "__main__":
    # quick smoke test
    rng = np.random.default_rng(0)
    T, H, W = 30, 50, 60
    truth = rng.normal(20, 2, (T, H, W)).astype(np.float32)
    # add a smooth trend so EOF modes can capture it
    for t in range(T):
        truth[t] += 0.05 * t
    valid = (rng.random((T, H, W)) > 0.3).astype(np.uint8)
    data = np.where(valid.astype(bool), truth, np.nan)

    filled, k, rmse = dineof_with_cv(data, valid, k_candidates=(2, 4, 6, 8))
    print(f"chosen k={k}, CV RMSE={rmse:.4f}")
    err = np.abs(filled - truth)[valid == 0]
    print(f"missing MAE={err.mean():.4f}, max={err.max():.4f}")
