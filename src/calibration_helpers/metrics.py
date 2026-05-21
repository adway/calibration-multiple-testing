import numpy as np
from scipy.integrate import quad

from .mixtures import (
    sample_mixture,
    mixture_density,
    conditional_mean,
)
from .calibrators import (
    PValueCalibrator,
    QValueCalibrator,
    GrenanderLfdrCalibrator,
    LindseyLfdrCalibrator,
    SplineMLELfdrCalibrator
)


def interval_integral(i, pi0, alpha, beta, knots, heights):
    def integrand(x):
        return mixture_density(x, pi0, alpha, beta) * (
            conditional_mean(x, pi0, alpha, beta) - heights[i]
        ) ** 2

    return quad(integrand, knots[i], knots[i + 1])[0]


def total_integral(pi0, alpha, beta, knots, heights):
    return sum(
        interval_integral(i, pi0, alpha, beta, knots, heights)
        for i in range(len(knots) - 1)
    )

def calibration_error(
    calibrator,
    pi0=0.90,
    alpha=0.5,
    beta=2.3,
    use_MC=False,
    B=100_000,
):
    if use_MC:
        samples = sample_mixture(B, pi0, alpha, beta)
        preds = calibrator.predict(samples)

        return np.mean(
            (conditional_mean(samples, pi0, alpha, beta) - preds) ** 2
        )

    if hasattr(calibrator, "knots_") and hasattr(calibrator, "heights_"):
        return total_integral(
            pi0,
            alpha,
            beta,
            calibrator.knots_,
            calibrator.heights_,
        )

    raise ValueError(
        "Analytic calibration error requires a level-set / step-function "
        "representation via `knots_` and `heights_`. "
        "Set `use_MC=True` for smooth/non-level-set calibrators."
    )


def fit_calibrator(name, pvals, labels=None, lam=0.5):
    if name == "p-value":
        cal = PValueCalibrator()
    elif name == "q-value":
        cal = QValueCalibrator(lam=lam)
    elif name == "g-lfdr":
        cal = GrenanderLfdrCalibrator()
    elif name == "l-lfdr":
        cal = LindseyLfdrCalibrator()
    elif name == "s-mle-lfdr":
        cal = SplineMLELfdrCalibrator()
    else:
        raise ValueError(f"Unknown calibrator: {name}")

    return cal.fit(pvals, labels)


def calibrate_and_assess(
    pvals,
    labels,
    lam=0.5,
    calibrator="p-value",
    pi0=0.90,
    alpha=0.5,
    beta=2.3,
    use_MC=False,
    B=100_000,
):
    cal = fit_calibrator(calibrator, pvals, labels, lam=lam)

    err = calibration_error(
        cal,
        pi0=pi0,
        alpha=alpha,
        beta=beta,
        use_MC=use_MC,
        B=B,
    )

    return err
