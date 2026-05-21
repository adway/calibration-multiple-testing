import numpy as np
from npeb import Grenander

from .misc import qvalues, lindsey_density_estimator, pseudo_spline_reg


class BaseCalibrator:
    name = "base"

    def fit(self, pvals, labels=None):
        self.pvals_ = np.asarray(pvals)
        self.labels_ = None if labels is None else np.asarray(labels)
        return self

    def predict(self, x):
        raise NotImplementedError


class PValueCalibrator(BaseCalibrator):
    name = "p-value"

    def predict(self, x):
        return np.asarray(x)


class QValueCalibrator(BaseCalibrator):
    name = "q-value"

    def __init__(self, lam=None):
        self.lam = lam

    def fit(self, pvals, labels=None):
        super().fit(pvals, labels)

        lam = self.lam
        if lam is None:
            lam = 1 - len(self.pvals_) ** (-1 / 5)

        self.lam_ = lam

        self.qvals_ = qvalues(self.pvals_, lam=lam)

        order = np.argsort(self.pvals_)

        self.knots_ = np.concatenate([
            [0.0],
            self.pvals_[order],
            [1.0],
        ])

        self.heights_ = np.concatenate([
            [self.qvals_[order][0]],
            self.qvals_[order],
            [1.0],
        ])

        return self

    def predict(self, x):
        x = np.asarray(x)

        idx = np.searchsorted(self.knots_, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.heights_) - 1)

        return self.heights_[idx]


class GrenanderLfdrCalibrator(BaseCalibrator):
    name = "g-lfdr"

    def __init__(self, lam=None, x_min=0.0, x_max=1.0):
        self.lam = lam
        self.x_min = x_min
        self.x_max = x_max

    def fit(self, pvals, labels=None):
        self.pvals_ = np.asarray(pvals)
        self.labels_ = None if labels is None else np.asarray(labels)

        self.gren_ = Grenander(x_min=self.x_min, x_max=self.x_max)
        self.gren_.fit(self.pvals_)

        lam = self.lam
        if lam is None:
            lam = 1 - len(self.pvals_) ** (-1 / 5)

        self.lam_ = lam
        self.pi0_hat_ = np.mean(self.pvals_ > lam) / (1 - lam)
        self.pi0_hat_ = min(self.pi0_hat_, 1.0)

        self.knots_ = self.gren_.knots
        self.heights_ = np.minimum(self.pi0_hat_ / self.gren_.slopes, 1.0)

        return self

    def predict(self, x):
        x = np.asarray(x)
        idx = np.searchsorted(self.knots_, x, side="right") - 1
        idx = np.clip(idx, 0, len(self.heights_) - 1)
        return self.heights_[idx]
    

class LindseyLfdrCalibrator(BaseCalibrator):
    name = "l-lfdr"

    def __init__(
        self,
        lam=None,
        num_bins=100,
        n_knots=10,
        degree=3,
        alpha=1e-6,
        grid_size=1000,
    ):
        self.lam = lam
        self.num_bins = num_bins
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        self.grid_size = grid_size

    def fit(self, pvals, labels=None):
        super().fit(pvals, labels)

        self.grid_, self.fhat_grid_, self.model_ = lindsey_density_estimator(
            self.pvals_,
            num_bins=self.num_bins,
            n_knots=self.n_knots,
            degree=self.degree,
            alpha=self.alpha,
            grid_size=self.grid_size,
        )

        lam = self.lam
        if lam is None:
            lam = 1 - len(self.pvals_) ** (-1 / 5)

        self.lam_ = lam
        self.pi0_hat_ = np.mean(self.pvals_ > lam) / (1 - lam)
        self.pi0_hat_ = min(self.pi0_hat_, 1.0)

        self.bin_width_ = 1 / self.num_bins

        return self

    def density(self, x):
        x = np.asarray(x)

        pred_counts = self.model_.predict(x.reshape(-1, 1))
        fhat = pred_counts / (len(self.pvals_) * self.bin_width_)

        # match your helper: numerically renormalize on grid
        area = np.trapezoid(self.fhat_grid_, self.grid_)
        return fhat / area

    def predict(self, x):
        fhat = self.density(x)
        return np.minimum(self.pi0_hat_ / fhat, 1.0)
    

class SplineMLELfdrCalibrator(BaseCalibrator):
    name = "s-mle-lfdr"

    def __init__(
        self,
        n_knots=20,
        degree=3,
        alpha=1e-6,
        eps=1e-12,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        self.eps = eps

    def fit(self, pvals, labels=None):
        super().fit(pvals, labels)

        self.fit_ = pseudo_spline_reg(
            self.pvals_,
            n_knots=self.n_knots,
            degree=self.degree,
            alpha=self.alpha,
            eps=self.eps,
        )

        self.pi0_hat_ = self.fit_["pi_hat_0"]
        self.lam_ = self.fit_["lambda"]

        return self

    def predict(self, x):
        return np.minimum(self.fit_["predict_g"](x), 1.0)