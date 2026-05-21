import numpy as np
import scipy.stats as stats


def make_pvals(n, pi0, alpha=0.5, beta=1.0):
    n0 = int(pi0 * n)
    n1 = n - n0

    pvals = np.concatenate([
        np.random.uniform(size=n0),
        np.random.beta(a=alpha, b=beta, size=n1),
    ])

    labels = np.concatenate([
        np.zeros(n0),
        np.ones(n1),
    ])

    perm = np.random.permutation(n)
    return labels[perm], pvals[perm]


def mixture_density(x, pi0, alpha, beta):
    return pi0 + (1 - pi0) * stats.beta.pdf(x, a=alpha, b=beta)


def sample_mixture(n, pi0, alpha, beta):
    n0 = int(pi0 * n)
    n1 = n - n0

    samples = np.concatenate([
        np.random.uniform(size=n0),
        np.random.beta(a=alpha, b=beta, size=n1),
    ])

    return samples[np.random.permutation(n)]


def conditional_mean(x, pi0, alpha, beta):
    return pi0 / mixture_density(x, pi0, alpha, beta)

