import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF, Matern, RationalQuadratic


def GP_functions_gen(kernel = RBF(), seq_len=100, samples = 1000, noise=False):

    gp = GaussianProcessRegressor(kernel=kernel)
    t = np.random.uniform(-2, 2,(samples, seq_len))

    # GP(0, RBF(1))
    Z_t = np.array([gp.sample_y(t[i, :].reshape(-1, 1), 1) for i in range(samples)])

    ϵ_t = 0
    # indep. noise
    if noise:
        ϵ_t = np.random.normal(0, 0.1, (samples,T//2, 1))

    ## Probability model
    Y_t = Z_t + ϵ_t

    return t, Y_t