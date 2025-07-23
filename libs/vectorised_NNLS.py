import numpy as np ## use import cupy as np for GPU, should be compatible but not tested
from scipy.optimize import nnls

def vectorised_lstsq(A, b):
    """
    Vectorised least squares solution for multiple linear regression."

    Parameters
    ----------
    A : array_like
        The design matrix of shape (..., m, n), where m is the number of observations and n is the number of predictors.
    b : array_like
        The response vector of shape (..., m).

    Returns
    -------
    x : array_like
        The estimated coefficients of shape (..., n).
    rss : array_like
        The residual sum of squares of shape (...).
    dof : array_like
        The degrees of freedom of shape (...).
    se : array_like
        The standard errors of the estimated coefficients of shape (..., n).
    """
    m = A.shape[-2]  # number of observations
    n = A.shape[-1]  # number of predictors
    dof = m - n

    ATA = np.einsum('...mi,...mj->...ij', A, A)
    ATb = np.einsum('...mi,...m->...i', A, b)
    x = np.linalg.solve(ATA, ATb)
    residuals = b - np.einsum('...ij,...j->...i', A, x)
    rss = np.sum(np.square(residuals), axis = -1)
    sigma2 = rss / dof
    ATA_inv = np.linalg.inv(ATA)
    diagATAinv = np.diagonal(ATA_inv, axis1 = -2, axis2 = -1)
    se = np.sqrt(sigma2[..., np.newaxis] * diagATAinv)

    return x, rss, dof, se

def ht(thetas, ts):
    """
    Calculate the h_t function in the lp-ntPET model.

    Parameters
    ----------
    thetas : array_like
        The parameters of shape (3, n), where n is the number of basis functions.
        The parameters are [alpha, tD, tP].
    ts : array_like
        The time points of shape (m,), where m is the number of time points.
        The time points are in seconds.

    Returns
    -------
    array_like
        The h_t function values of shape (n, m).
        The h_t function is defined as:
        h_t = max(0, (t - tD) / (tP - tD)) ^ alpha * exp(alpha * (1 - (t - tD) / (tP - tD))) * (t > tD)
    """

    alphas, tDs, tPs = thetas
    ts = ts[np.newaxis, ...]
    alphas = alphas[..., np.newaxis]
    tDs = tDs[..., np.newaxis]
    tPs = tPs[..., np.newaxis]
    return np.maximum(0, (ts - tDs) / (tPs - tDs)) ** alphas * np.exp(alphas * (1 - (ts - tDs) / (tPs - tDs))) * (ts > tDs)

def get_optimal_basis(rss, thetas, params, dof, se):
    """
    Get the optimal basis for the lp-ntPET model.
    The optimal basis is the one that minimizes the residual sum of squares (rss).
    The optimal parameters are the ones that correspond to the minimum rss.
    The optimal parameters are returned as a concatenation of the optimal parameters and thetas.
    The optimal standard errors are also returned.
    The optimal standard errors are the ones that correspond to the minimum rss.

    Parameters
    ----------
    rss : array_like
        The residual sum of squares of shape (n, m), where n is the number of basis functions and m is the number of voxels.
    thetas : array_like
        The parameters of shape (q, n) from the basis function, where n is the number of basis function and q is the number of parameters.
        The parameters are [alpha, tD, tP].
    params : array_like
        The parameters of shape (n, m, p), where n is the number of parameters, m is the number of voxels and p is the number of parameters.
        The parameters are [R1, k2, k2a, gamma].
    dof : array_like
        The degrees of freedom, scalar.
    se : array_like
        The standard errors of the estimated coefficients of shape (n, m, p), where n is the number of parameters, m is the number of voxels and p is the number of parameters.
        The standard errors are [R1, k2, k2a, gamma].

    Returns
    -------
    return_params : array_like
        The optimal parameters of shape (m, p + 3), where m is the number of voxels and p is the number of parameters.
        The optimal parameters are [R1, k2, k2a, gamma, alpha, tD, tP].
    min_rss : array_like
        The minimum residual sum of squares of shape (m,), where m is the number of voxels.
    min_idx : array_like
        The index of the minimum residual sum of squares, scalar.
    optimal_se : array_like
        The optimal standard errors of the estimated coefficients of shape (m, p), where m is the number of voxels and p is the number of parameters.
        The optimal standard errors are [R1, k2, k2a, gamma].
    """
    min_rss = np.min(rss, axis = 0)
    min_idx = np.argmin(rss, axis = 0)
    optimal_thetas = thetas.T[min_idx]
    optimal_params = params[min_idx, np.arange(params.shape[1]), :]
    optimal_se = se[min_idx, np.arange(se.shape[1]), :]
    return_params = np.concatenate((optimal_params, optimal_thetas), axis = -1)
    return return_params, min_rss, dof, optimal_se

def get_BIC(rss, n_params, n_samples):
    """"
    Calculate the Bayesian Information Criterion (BIC) for the lp-ntPET model."

    Parameters
    ----------
    rss : array_like
        The residual sum of squares of shape (...).
    n_params : int
        The number of parameters in the model.
    n_samples : int
        The number of samples in the model.
        
    Returns
    -------
    float
        The BIC value.
    """
    p = n_params
    n = n_samples
    return n * np.log(rss / n) + p * np.log(n)

def nnls_once(A, b):
    # your NNLS call
    x, rnorm = nnls(A, b)

    # 1. RSS
    rss = rnorm**2

    # 2. Identify active set (those x_j > 0)
    tol_active = 1e-8
    J = np.where(x > tol_active)[0]

    # 3. Degrees of freedom
    n, p = A.shape
    dof = n - p

    # 4. Estimate σ²
    sigma2 = rss / dof

    # 5. Compute SE for each active coefficient
    A_J = A[:, J]
    # (A_J^T A_J) must be invertible; if it isn’t, you’ll need to regularize or bootstrap
    cov_active = sigma2 * np.linalg.inv(A_J.T @ A_J)
    se = np.zeros_like(x)
    se[J] = np.sqrt(np.diag(cov_active))

    return x, rss, dof, se

def nnls_stats(As: np.ndarray,
               bs: np.ndarray):
    """
    Sadly, this cannot be vectorised
    As the nnls does not have a closed form result
    It requires a convex optimisation under convex constraints
    """
    if bs.ndim == 1:
        return nnls_once(As, bs)
    
    if As.ndim - bs.ndim == 1:
        xs = []
        rsss = []
        ses = []
        for i in range(As.shape[0]):
            x, rss, dof, se = nnls_once(As[i], bs[i])
            xs.append(x)
            rsss.append(rss)
            ses.append(se)
        return np.array(xs), np.array(rsss), dof, np.array(ses)
    
    if As.ndim - bs.ndim == 2:
        xss = []
        rssss = []
        sess = []
        for i in range(As.shape[0]):
            xs = []
            rsss = []
            ses = []
            for j in range(As.shape[1]):
                x, rss, dof, se = nnls_once(As[i, j], bs[j])
                xs.append(x)
                rsss.append(rss)
                ses.append(se)
            xss.append(xs)
            rssss.append(rsss)
            sess.append(ses)
        return np.array(xss), np.array(rssss), dof, np.array(sess)
    else:
        raise ValueError("Dimension is wrong!")
        # return None, None, None, None

def get_curve(Ct, Cr, t, thetas):

    R1, k2, k2a, gamma, alpha, tD, tP = thetas

    Ct_cumsum = np.cumsum(Ct)
    Cr_cumsum = np.cumsum(Cr)
    dt = t[1] - t[0]

    if np.isnan(gamma):
        return R1 * Cr + k2 * Cr_cumsum * dt - k2a * Ct_cumsum * dt
    else:
        ht = np.maximum(0, (t - tD) / (tP - tD)) ** alpha * np.exp(alpha * (1 - (t - tD) / (tP - tD))) * (t > tD)
        Bts = np.cumsum(Ct * ht)
        return R1 * Cr + k2 * Cr_cumsum * dt - k2a * Ct_cumsum * dt - gamma * Bts * dt