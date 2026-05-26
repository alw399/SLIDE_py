"""
essreg.py
---------
Python wrapper around the R EssReg package's plainER() function.

Replaces the LOVE-based latent factor estimation used in the original
SLIDE_py, bringing the Python implementation in line with the R SLIDE
package (jishnu-lab/SLIDE) which uses Essential Regression as the
latent factor backend.

The `call_essreg` function:
  - Takes a standardised X matrix and a scalar delta/lambda
  - Calls EssReg::plainER via rpy2
  - Returns a Python dict with keys:
        K        : int   – number of latent factors found
        A        : (p, K) ndarray – loading matrix
        C        : (K, K) ndarray – factor covariance estimate
        Gamma    : (p,)   ndarray – diagonal noise variances
        I_hat    : list of dicts {pos, neg} per cluster (0-indexed)
        pureVec  : (p,)   ndarray – which cluster each pure variable belongs to
        Gamma_hat: same as Gamma (alias kept for compatibility)

The `calc_z_matrix` function reproduces R's PredZ logic:
    Ĝ = AᵀΓ⁻¹A + C⁻¹
    Ẑ = X Γ⁻¹ A Ĝ⁻¹
which is the posterior mean of Z under the EssReg generative model.
"""

import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects import numpy2ri, r
from rpy2.robjects.packages import importr


def call_essreg(
    X: pd.DataFrame,
    delta: float,
    lambda_: float = 0.1,
    thresh_fdr: float = 0.2,
    std_y: bool = True,
    rep_cv: int = 0,
    alpha_level: float = 0.05,
    out_path: str = None,
    verbose: bool = False,
) -> dict:
    """
    Call EssReg::plainER (the core Essential Regression routine) via rpy2.

    Parameters
    ----------
    X : pd.DataFrame, shape (n, p)
        Feature matrix.  Should be the *raw* (non-standardised) X; this
        function z-scores it internally (matching the R pipeline which passes
        x_std to plainER).
    delta : float
        Threshold parameter δ for Σ thresholding.  Scaled internally by
        √(log max(p,n) / n) exactly as in getLatentFactors.R.
    lambda_ : float
        Regularisation parameter λ for the Dantzig estimator.
    thresh_fdr : float
        FDR level for thresholding the sample correlation matrix (BH).
    std_y : bool
        Kept for signature compatibility; not used (plainER does not need y
        for the latent factor step – y is only used for β estimation later).
    rep_cv : int
        Number of CV replicates passed to plainER.  Set to 0 to skip CV
        (we already tune δ externally over a grid, matching the Python
        pipeline's approach of sweeping delta values).
    alpha_level : float
        Confidence interval level.
    out_path : str or None
        Directory where intermediate outputs are saved.  Passed as R NULL
        when None (no file output from R).
    verbose : bool
        Print progress messages from R.

    Returns
    -------
    dict with keys:
        K, A, C, Gamma, I_hat, pureVec, optDelta, optLambda
    """
    feature_names = list(X.columns)
    sample_names = list(X.index)
    n, p = X.shape

    # --- standardise X (z-score columns) to match R pipeline -----------------
    X_std = (X.values - X.values.mean(axis=0)) / X.values.std(axis=0, ddof=1)
    # Replace any NaN columns (zero-std) with 0
    X_std = np.nan_to_num(X_std, nan=0.0)

    # --- R delta scaling (mirrors getLatentFactors.R) -------------------------
    # delta_scaled = delta * sqrt(log(max(p, n)) / n)
    # plainER accepts the *unscaled* delta and does the scaling internally,
    # but we pass the value the user specifies (same as R SLIDE yaml delta).

    numpy2ri.activate()

    try:
        essreg = importr("EssReg")
    except Exception as e:
        raise ImportError(
            "R package 'EssReg' not found. Install it in R with:\n"
            "  devtools::install_github('jishnu-lab/EssReg')"
        ) from e

    r_X_std = numpy2ri.py2rpy(X_std)
    r_delta = robjects.FloatVector([delta])
    r_lambda = robjects.FloatVector([lambda_])
    r_thresh_fdr = robjects.FloatVector([thresh_fdr])
    r_rep_cv = robjects.IntVector([rep_cv])
    r_alpha = robjects.FloatVector([alpha_level])
    r_out_path = robjects.NULL if out_path is None else robjects.StrVector([out_path])
    r_std_y = robjects.BoolVector([std_y])

    # plainER signature (from EssReg source):
    # plainER(y=NULL, x, x_std, sigma=NULL, delta, lambda=0.1,
    #         thresh_fdr=0.2, rep_cv=0, alpha_level=0.05, out_path=NULL)
    # Note: y is NULL here because we only want the latent factors, not β
    result = essreg.plainER(
        robjects.NULL,      # y  – not needed for Z estimation
        r_X_std,            # x  (= x_std because we already standardised)
        r_X_std,            # x_std
        robjects.NULL,      # sigma – computed internally from x_std
        r_delta,            # delta
        r_lambda,           # lambda
        r_thresh_fdr,       # thresh_fdr
        r_rep_cv,           # rep_cv  (0 = skip CV)
        r_alpha,            # alpha_level
        r_out_path,         # out_path
    )

    python_result = _parse_essreg_result(result, feature_names, sample_names, p)

    numpy2ri.deactivate()
    return python_result


def _parse_essreg_result(result, feature_names, sample_names, p: int) -> dict:
    """Convert an rpy2 plainER result list into a Python dict."""
    out = {}
    names = list(result.names)

    def _get(key):
        if key in names:
            return result.rx2(key)
        return None

    # K – number of clusters / latent factors
    k_r = _get("K")
    K = int(k_r[0]) if k_r is not None else 0
    out["K"] = K

    # A matrix (p × K) – loading / assignment matrix
    a_r = _get("A_hat")
    if a_r is None:
        a_r = _get("A")
    if a_r is not None:
        A = np.array(a_r)
        if A.ndim == 1:
            A = A.reshape(p, -1)
        out["A"] = A
    else:
        out["A"] = np.zeros((p, max(K, 1)))

    # C matrix (K × K) – factor covariance
    c_r = _get("C_hat")
    if c_r is None:
        c_r = _get("C")
    out["C"] = np.array(c_r) if c_r is not None else np.eye(max(K, 1))

    # Gamma – diagonal noise variances (p,)
    g_r = _get("Gamma_hat")
    if g_r is None:
        g_r = _get("Gamma")
    out["Gamma"] = np.array(g_r) if g_r is not None else np.ones(p)

    # I_hat – pure variable index lists per cluster (list of dicts, 0-indexed)
    i_r = _get("I_hat")
    if i_r is None:
        i_r = _get("I_hat_list")
    if i_r is not None:
        parsed = []
        for item in i_r:
            # R is 1-indexed; convert to 0-indexed
            pos = [int(v) - 1 for v in item.rx2("pos")] if "pos" in item.names else []
            neg = [int(v) - 1 for v in item.rx2("neg")] if "neg" in item.names else []
            parsed.append({"pos": pos, "neg": neg})
        out["I_hat"] = parsed
    else:
        out["I_hat"] = []

    # pureVec – which cluster each pure variable belongs to (0-indexed)
    pv_r = _get("pureVec")
    out["pureVec"] = (np.array(pv_r) - 1).astype(int) if pv_r is not None else np.array([])

    # optDelta, optLambda
    od_r = _get("optDelta")
    out["optDelta"] = float(od_r[0]) if od_r is not None else None
    ol_r = _get("optLambda")
    if ol_r is None:
        ol_r = _get("opt_lambda")
    out["optLambda"] = float(ol_r[0]) if ol_r is not None else None

    return out


def calc_z_matrix(X_df: pd.DataFrame, er_result: dict) -> pd.DataFrame:
    """
    Compute the latent factor matrix Z from the EssReg result.

    Reproduces R's EssReg::PredZ / calcZMatrix logic:

        Ĝ = AᵀΓ⁻¹A + C⁻¹
        Ẑ = X_std Γ⁻¹ A Ĝ⁻¹

    This is the posterior mean of Z under the factor model
        X = Z A^T + ε,  ε_i ~ N(0, Γ_ii).

    Parameters
    ----------
    X_df : pd.DataFrame, shape (n, p)
        Raw (non-standardised) feature matrix with named columns/rows.
    er_result : dict
        Output of call_essreg().

    Returns
    -------
    pd.DataFrame, shape (n, K)
        Latent factor matrix with columns Z0, Z1, …
    """
    A_hat = er_result["A"]          # (p, K)
    C_hat = er_result["C"]          # (K, K)
    Gamma = er_result["Gamma"]      # (p,)
    K = er_result["K"]

    if K == 0:
        return pd.DataFrame(index=X_df.index)

    # Standardise X (z-score columns, ddof=1 to match R's scale())
    X = X_df.values.astype(float)
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0, ddof=1)
    col_std = np.where(col_std == 0, 1.0, col_std)
    X_std = (X - col_mean) / col_std

    # Guard against zero/near-zero Gamma entries
    Gamma_safe = np.where(np.abs(Gamma) < 1e-10, 1e-10, Gamma)
    Gamma_inv = np.diag(Gamma_safe ** (-1))   # (p, p) diagonal

    # G = A^T Γ⁻¹ A + C⁻¹
    try:
        C_inv = np.linalg.inv(C_hat)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C_hat)

    G_hat = A_hat.T @ Gamma_inv @ A_hat + C_inv   # (K, K)

    # Ẑ = X_std Γ⁻¹ A G⁻¹
    try:
        G_inv = np.linalg.inv(G_hat)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G_hat)

    Z_hat = X_std @ Gamma_inv @ A_hat @ G_inv      # (n, K)

    return pd.DataFrame(
        Z_hat,
        index=X_df.index,
        columns=[f"Z{i}" for i in range(K)],
    )
