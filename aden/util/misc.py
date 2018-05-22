__author__ = ["jeremiah"]

# misc functions
import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def array_cosine(w_1, w_2):
    nom = np.sum(w_1 * w_2, axis=1)
    denom = np.linalg.norm(w_1, axis=1) * np.linalg.norm(w_2, axis=1)
    return nom/denom


def spatial_weight_gen(X, K, ls=1., n_induce=[50, 50], visual=False):
    """
    generate spatially varying sparse weight using Sparse GP method with RBF kernel

    NOTE:
    f_induce and f_interpol are standardized (by row sum of kernel matrix) so
    the generated weights are roughly on the same scale

    :param X: input data points, dim N x P
    :param K: number of base models
    :param n_induce: list of number of inducing points (for each dimension of X) for sparse GP.
    :param ls: length scale parameter for RBF kernel
    :return:
    """
    # select inducing points based on quantile
    N, P = X.shape
    N_induce = np.product(n_induce)

    assert len(n_induce) == P, \
        "n_induce need to be specified for each dimension of X"

    X_induce_marginal = \
        [np.percentile(X[:, index], q=np.linspace(0, 1, n_sample+2)[1:-1]*100)
         for index, n_sample in enumerate(n_induce)]
    X_induce = np.array(list(itertools.product(*X_induce_marginal)))

    # sample from inducing GP
    K_induce = rbf_kernel(X=X_induce, gamma=1/float(ls))
    alpha_induce = np.random.normal(size=(N_induce, K))
    print("standardizing f_induce by kernel row sum..")
    f_induce = K_induce.dot(alpha_induce)/np.sum(K_induce, axis=1)[:, None]

    # interpolate to full dataset
    K_interpol = rbf_kernel(X=X, Y=X_induce, gamma=1/float(ls))
    print("standardizing f_interpol by sqrt of kernel row sum..")
    f_interpol = K_interpol.dot(f_induce)/np.sqrt(np.sum(K_interpol, axis=1))[:, None]

    # produce weight, then return
    w_loc = np.exp(f_interpol)/np.sum(np.exp(f_interpol), axis=1)[:, None]

    if visual:
        # optionally, visualize
        loc_x, loc_y = np.meshgrid(
            np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), num=100),
            np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), num=100))
        loc_plot = np.c_[np.ravel(loc_x), np.ravel(loc_y)]
        K_plot = rbf_kernel(X=loc_plot, Y=X_induce, gamma=1 / float(ls))
        f_plot = K_plot.dot(f_induce)/np.sqrt(np.sum(K_plot, axis=1))[:, None]
        w_plot = np.exp(f_plot) / np.sum(np.exp(f_plot), axis=1)[:, None]
        loc_z = w_plot[:, 0].reshape(loc_x.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(loc_x, loc_y, loc_z, cmap=cm.coolwarm,
                               linewidth=0.1, antialiased=True)
        ax.set_zlim(np.min(loc_z), np.min(loc_z)+0.3)

    return w_loc


def add_dim_name(error_arr, row_name, col_name):
    row_name = np.array(row_name)[:, None]
    col_name = np.array([0] + col_name)[None, :]

    error_arr = np.concatenate((row_name, error_arr), axis=1)
    error_arr = np.concatenate((col_name, error_arr), axis=0)

    return error_arr