# %% kNN LDC utility functions


import numpy as np

import scipy.stats as st


# %% kNN with Local Density Corrections - utility function


def L2_norm(x):
    return np.sqrt(x.T.dot(x))


def pair_likelihood_pdf(query, pos_sample, neg_sample):
    # Returns a normal pdf
    # The weight of the pdf is closer to zero the more positive the query is

    t = query.reshape(-1, 1)
    xi = pos_sample.reshape(-1, 1)
    xj = neg_sample.reshape(-1, 1)

    yi = xi - t
    yj = xj - t
    xij = xi - xj

    # x_ij length
    d = L2_norm(xij)

    yi_ = xi.T.dot(xij) / d
    yj_ = xj.T.dot(xij) / d

    # Distance from x_ij
    l = np.sqrt(yi.T.dot(yi) - yi_.T.dot(yi_))

    # Scaled down by d
    mu = yi_ / d

    alpha = 1.0
    scale = alpha * np.sqrt(l/d)

    return st.norm(mu, scale)


def get_point_membership_distribution(query, pos_samples, neg_samples):
    return [
        pair_likelihood_pdf(query,
                            pos_samples[pos_sample_inx, :].T,
                            neg_samples[neg_sample_inx, :].T)
        for neg_sample_inx in np.arange(neg_samples.shape[0])
        for pos_sample_inx in np.arange(pos_samples.shape[0])
    ]


def get_weight(t, pos_samples, neg_samples):
    if len(neg_samples) > 5:
        neg_samples = neg_samples[:5]

    point_membership_distribution =\
        get_point_membership_distribution(t, pos_samples, neg_samples)

    # return np.mean([max(0, min(1, d)) for d in point_membership_distribution])
    lower_bound = min([pdf.ppf(0.01)
                       for pdf in point_membership_distribution])[0]
    upper_bound = max([pdf.ppf(0.99)
                       for pdf in point_membership_distribution])[0]

    if np.isnan(lower_bound) or np.isnan(upper_bound):
        return 0.5

    def eval_pdfs(x):
        return np.sum([pdf.pdf(x) for pdf in point_membership_distribution])

    xs = np.linspace(lower_bound, upper_bound, num=100)
    ys = np.array([eval_pdfs(x) for x in xs])
    ys_pdf = ys / np.sum(ys)
    ys_cdf = np.cumsum(ys_pdf)

    def get_percentile(xs, ys_cdf, percent):
        mask = ys_cdf < percent
        tmp = np.where(mask)[0]
        if len(tmp) == 0:
            return 0.5
        last_true = tmp[-1]
        return xs[last_true]

    return max(0.0, min(1.0, get_percentile(xs, ys_cdf, 0.5)))

# %%


xi = np.array([[0], [3]])
xj = np.array([[5], [2]])
t = np.array([[0], [0]])

p = pair_likelihood_pdf(query=t, pos_sample=xi, neg_sample=xj)
