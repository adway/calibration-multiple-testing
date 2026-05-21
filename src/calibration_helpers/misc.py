import numpy as np
from scipy.optimize import isotonic_regression, minimize
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline


def qvalues(pvals, lam=0.5):
    pvals = np.asarray(pvals)
    m = len(pvals)

    order = np.argsort(pvals)
    p_sorted = pvals[order]

    pi0_hat = np.mean(pvals > lam) / (1 - lam)
    pi0_hat = min(pi0_hat, 1.0)

    q_sorted = m * p_sorted * pi0_hat / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    q_original = np.empty_like(q_sorted)
    q_original[order] = q_sorted

    return q_original


# Lindsey's method for density estimation and lfdr estimation
def lindsey_density_estimator(
    pvals,
    num_bins=100,
    n_knots=12,
    degree=3,
    alpha=1e-6,
    grid_size=1000,
):
    pvals = np.asarray(pvals)
    pvals = pvals[(pvals >= 0) & (pvals <= 1)]

    n = len(pvals)

    # Histogram counts, not density
    counts, bin_edges = np.histogram(
        pvals,
        bins=num_bins,
        range=(0, 1),
        density=False
    )

    h = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    X = bin_centers.reshape(-1, 1)
    y = counts

    # Spline Poisson regression:
    # log E[N_j] = spline(x_j)
    model = make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            include_bias=False
        ),
        PoissonRegressor(
            alpha=alpha,
            max_iter=5000
        )
    )

    model.fit(X, y)

    # Evaluate on fine grid
    grid = np.linspace(0, 1, grid_size)
    pred_counts = model.predict(grid.reshape(-1, 1))

    # Convert predicted bin counts to density scale
    raw_density = pred_counts / (n * h)

    # Renormalize numerically so density integrates to 1
    area = np.trapezoid(raw_density, grid)
    density = raw_density / area

    return grid, density, model

def pseudo_spline_reg(
    p_vals,
    n_knots=20,
    degree=3,
    alpha=1e-6,
    eps=1e-12,
):
    # sort p-values
    p_vals = np.asarray(p_vals)
    p_vals = np.sort(p_vals)

    # add 0 and compute spacings
    p_aug = np.concatenate(([0.0], p_vals))
    diffs = np.diff(p_aug)

    # Storey pi0 estimate
    m = len(p_vals)
    lam = 1 - m ** (-1 / 5)
    pi_hat_0 = np.mean(p_vals > lam) / (1 - lam)

    # pseudo-labels
    Y = pi_hat_0 * m * diffs
    Y = np.maximum(Y, eps)

    # spline basis evaluated at observed p-values
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=True
    )

    X = spline.fit_transform(p_vals.reshape(-1, 1))

    # initialize log g near log Y
    theta0, *_ = np.linalg.lstsq(X, np.log(Y), rcond=None)

    def objective(theta):
        eta = X @ theta
        g = np.exp(eta)

        ratio = Y / g

        loss = np.mean(ratio - np.log(ratio) - 1)

        # light ridge penalty to stabilize spline
        penalty = alpha * np.sum(theta[1:] ** 2)

        return loss + penalty

    res = minimize(
        objective,
        theta0,
        method="BFGS"
    )

    theta_hat = res.x

    def predict_g(x):
        x = np.asarray(x)
        X_new = spline.transform(x.reshape(-1, 1))
        return np.exp(X_new @ theta_hat)

    return {
        "theta": theta_hat,
        "spline": spline,
        "predict_g": predict_g,
        "p_vals": p_vals,
        "Y": Y,
        "pi_hat_0": pi_hat_0,
        "lambda": lam,
        "result": res,
    }